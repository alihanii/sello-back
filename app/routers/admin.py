from datetime import datetime, time, timedelta, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.deps import DB, AdminUser
from app.models import (
    AI_TASKS, IMAGE_KINDS, IMAGE_TASKS, AdminAuditLog, AiProvider, AiTaskRoute, AiUsageLog, Channel, ChannelListing, FeatureInterest,
    Operation, Payment, Plan, PlanRequest, Product, Store, Subscription, User,
)
from app.services.bank import BANKS
from app.services.text import to_en_digits
from app.services.features import FEATURES, MODES, allowed, feature_access, save_feature_access
from app.services.integrations import (
    INSTAGRAM_SCOPES, instagram_app, instagram_crawler, payment_config, save_instagram_app, save_instagram_crawler,
    save_payment_config,
)
from app.services.plans import (
    DURATIONS, FLAG_KEYS, LIMIT_KEYS, all_tiers, chat_used, create_subscription, credits_used, free_limits, limits_for, plan_status,
    save_free_limits, tier_config,
)
from app.schemas import ChannelOut, PlanOut, PlanTier, StoreOut, UserOut
from app.services.ai import DEFAULT_BASE_URLS, AIError, check_model, list_models, metis_family, model_kind, provider_proxy, test_provider

router = APIRouter(prefix="/admin", tags=["admin"])


def audit(db, admin: User, action: str, target_type: str, target_id: int | None, diff: dict | None = None) -> None:
    db.add(AdminAuditLog(admin_id=admin.id, action=action, target_type=target_type, target_id=target_id, diff=diff or {}))


# ---------- overview ----------

@router.get("/stats")
async def stats(admin: AdminUser, db: DB):
    now = utcnow()
    count = lambda m, *w: db.scalar(select(func.count()).select_from(m).where(*w))  # noqa: E731
    by_platform = dict((await db.execute(select(Channel.platform, func.count()).where(Channel.status != "disconnected")
                                         .group_by(Channel.platform))).all())
    active_subs = (await db.execute(select(Subscription).where(Subscription.status == "active", Subscription.ends_at > now))).scalars().all()
    expiring = [s for s in active_subs if s.ends_at.replace(tzinfo=timezone.utc) < now + timedelta(days=7)]
    revenue = await db.scalar(select(func.coalesce(func.sum(Plan.price_toman), 0)).join(Subscription, Subscription.plan_id == Plan.id)
                              .where(Subscription.created_at >= now - timedelta(days=30), Subscription.activated_by != "trial"))
    return {
        "users": await count(User), "active_users_7d": await count(User, User.last_login_at >= now - timedelta(days=7)),
        "stores": await count(Store, ~Store.archived), "channels_by_platform": by_platform,
        "products": await count(Product, Product.status != "archived"),
        "publishes_30d": await count(Operation, Operation.type == "publish", Operation.status == "succeeded",
                                     Operation.created_at >= now - timedelta(days=30)),
        "active_subscriptions": len(active_subs), "expiring_7d": len(expiring), "revenue_30d": revenue,
        "pending_requests": await count(PlanRequest, PlanRequest.status == "pending"),
        "chat_waitlist": await count(FeatureInterest, FeatureInterest.feature == "ai_chat"),
    }


# ---------- users ----------

@router.get("/users")
async def users(admin: AdminUser, db: DB, q: str = ""):
    stmt = select(User).order_by(User.id.desc())
    if q:
        stmt = stmt.where((User.phone.contains(q)) | (User.name.contains(q)))
    rows = (await db.execute(stmt.limit(500))).scalars().all()
    stores = dict((await db.execute(select(Store.owner_id, func.count()).where(~Store.archived).group_by(Store.owner_id))).all())
    now = utcnow()
    subs = {s.user_id: s for s in (await db.execute(select(Subscription).where(Subscription.status == "active",
                                                                               Subscription.ends_at > now))).scalars()}
    out = []
    for u in rows:
        s = subs.get(u.id)
        plan = await db.get(Plan, s.plan_id) if s else None
        status = "admin" if u.role == "admin" else "blocked" if not u.is_active else "active" if s else None
        out.append({**UserOut.model_validate(u).model_dump(), "is_active": u.is_active, "stores": stores.get(u.id, 0),
                    "plan": plan.name if plan else None, "plan_tier": plan.tier if plan else None,
                    "plan_ends_at": s.ends_at if s else None, "plan_status": status or await plan_status(db, u)})
    return out


@router.get("/users/{user_id}")
async def user_detail(user_id: int, admin: AdminUser, db: DB):
    u = await db.get(User, user_id)
    if not u:
        raise HTTPException(404, "کاربر پیدا نشد.")
    stores = (await db.execute(select(Store).where(Store.owner_id == u.id))).scalars().all()
    store_rows = []
    for s in stores:
        channels = (await db.execute(select(Channel).where(Channel.store_id == s.id, Channel.status != "disconnected"))).scalars().all()
        products = await db.scalar(select(func.count()).select_from(Product).where(Product.store_id == s.id, Product.status != "archived"))
        publishes = await db.scalar(select(func.count()).select_from(Operation).where(
            Operation.store_id == s.id, Operation.type == "publish", Operation.status == "succeeded"))
        store_rows.append({**StoreOut.model_validate(s).model_dump(), "products": products, "publishes": publishes,
                           "channels": [ChannelOut.model_validate(c).model_dump() for c in channels]})
    subs = (await db.execute(select(Subscription).where(Subscription.user_id == u.id).order_by(Subscription.id.desc()))).scalars().all()
    sub_rows = []
    for s in subs:
        p = await db.get(Plan, s.plan_id)
        sub_rows.append({"id": s.id, "plan": p.name if p else "?", "plan_id": s.plan_id, "tier": p.tier if p else "",
                         "duration_months": p.duration_months if p else None,
                         "status": s.status, "starts_at": s.starts_at, "ends_at": s.ends_at, "activated_by": s.activated_by,
                         "note": s.note})
    features = {}
    for key, label in FEATURES.items():
        acc = await feature_access(db, key)
        features[key] = {"label": label, "mode": acc["mode"], "selected": u.id in acc["user_ids"], "allowed": allowed(acc, u)}
    limits = await limits_for(db, u)
    usage = {"chat_messages": await chat_used(db, u.id), "ai_credits": await credits_used(db, u.id),
             "stores": sum(1 for s in stores if not s.archived), "products": sum(r["products"] for r in store_rows),
             "channels": sum(len(r["channels"]) for r in store_rows)}
    return {"user": {**UserOut.model_validate(u).model_dump(), "is_active": u.is_active}, "stores": store_rows,
            "subscriptions": sub_rows, "features": features, "plan_status": await plan_status(db, u),
            "limits": limits, "usage": usage}


class ActivateIn(BaseModel):
    plan_id: int
    starts_at: datetime | None = None
    note: str = ""
    # "after": starts when the current plan ends (renewal); "now": ends the current plan and starts this one today.
    mode: str = "after"


async def _activate(db, admin: User, user_id: int, plan_id: int, starts_at: datetime | None, note: str,
                    replace: bool = False) -> Subscription:
    plan = await db.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "پلن پیدا نشد.")
    sub = await create_subscription(db, user_id, plan, activated_by=f"admin:{admin.id}", note=note, starts_at=starts_at,
                                    replace=replace)
    audit(db, admin, "change_plan" if replace else "activate_plan", "user", user_id,
          {"plan_id": plan.id, "starts_at": sub.starts_at.isoformat()})
    return sub


@router.post("/users/{user_id}/subscriptions")
async def activate(user_id: int, body: ActivateIn, admin: AdminUser, db: DB):
    if not await db.get(User, user_id):
        raise HTTPException(404, "کاربر پیدا نشد.")
    if body.mode not in ("after", "now"):
        raise HTTPException(422, "حالت نامعتبر است.")
    sub = await _activate(db, admin, user_id, body.plan_id, body.starts_at, body.note, replace=body.mode == "now")
    await db.commit()
    return {"ok": True, "ends_at": sub.ends_at}


@router.post("/subscriptions/{sub_id}/cancel")
async def cancel_sub(sub_id: int, admin: AdminUser, db: DB):
    s = await db.get(Subscription, sub_id)
    if not s:
        raise HTTPException(404, "اشتراک پیدا نشد.")
    s.status = "cancelled"
    audit(db, admin, "cancel_subscription", "subscription", sub_id)
    await db.commit()
    return {"ok": True}


class ActiveIn(BaseModel):
    is_active: bool


@router.post("/users/{user_id}/active")
async def set_active(user_id: int, body: ActiveIn, admin: AdminUser, db: DB):
    u = await db.get(User, user_id)
    if not u or u.id == admin.id:
        raise HTTPException(400, "این کاربر قابل تغییر نیست.")
    u.is_active = body.is_active
    audit(db, admin, "set_active", "user", user_id, {"is_active": body.is_active})
    await db.commit()
    return {"ok": True}


# ---------- features opened to a segment of sellers ----------


def _feature_key(key: str) -> str:
    if key not in FEATURES:
        raise HTTPException(404, "این قابلیت وجود ندارد.")
    return key


async def _feature_out(db, key: str) -> dict:
    acc = await feature_access(db, key)
    users = (await db.execute(select(User).where(User.id.in_(acc["user_ids"])).order_by(User.id.desc()))).scalars().all() \
        if acc["user_ids"] else []
    sellers = await db.scalar(select(func.count()).select_from(User).where(User.role != "admin", User.is_active))
    return {"key": key, "label": FEATURES[key], "mode": acc["mode"],
            "users": [{"id": x.id, "phone": x.phone, "name": x.name} for x in users],
            "sellers": sellers or 0,
            "with_access": (sellers or 0) if acc["mode"] == "all" else len(users) if acc["mode"] == "selected" else 0}


@router.get("/features")
async def features(admin: AdminUser, db: DB):
    return [await _feature_out(db, k) for k in FEATURES]


class FeatureIn(BaseModel):
    mode: str
    user_ids: list[int] | None = None  # None keeps the current list
    phones: list[str] = []  # add users by phone too


@router.put("/features/{key}")
async def put_feature(key: str, body: FeatureIn, admin: AdminUser, db: DB):
    key = _feature_key(key)
    if body.mode not in MODES:
        raise HTTPException(422, "حالت نامعتبر است.")
    ids = list(body.user_ids) if body.user_ids is not None else (await feature_access(db, key))["user_ids"]
    for phone in body.phones:
        phone = to_en_digits(phone).strip()
        u = (await db.execute(select(User).where(User.phone == phone))).scalar_one_or_none()
        if not u:
            raise HTTPException(404, f"کاربری با شمارهٔ {phone} پیدا نشد.")
        ids.append(u.id)
    v = await save_feature_access(db, key, body.mode, ids)
    audit(db, admin, "update_feature", "feature", None, {"key": key, **v})
    await db.commit()
    return await _feature_out(db, key)


class UserFeatureIn(BaseModel):
    enabled: bool


@router.post("/users/{user_id}/features/{key}")
async def set_user_feature(user_id: int, key: str, body: UserFeatureIn, admin: AdminUser, db: DB):
    key = _feature_key(key)
    if not await db.get(User, user_id):
        raise HTTPException(404, "کاربر پیدا نشد.")
    acc = await feature_access(db, key)
    ids = set(acc["user_ids"]) | {user_id} if body.enabled else set(acc["user_ids"]) - {user_id}
    v = await save_feature_access(db, key, acc["mode"], list(ids))
    audit(db, admin, "update_feature", "user", user_id, {"key": key, "enabled": body.enabled})
    await db.commit()
    return {"ok": True, "mode": v["mode"], "selected": body.enabled}


# ---------- stores & channels ----------

@router.get("/stores")
async def all_stores(admin: AdminUser, db: DB):
    rows = (await db.execute(select(Store, User).join(User, User.id == Store.owner_id).order_by(Store.id.desc()))).all()
    out = []
    for s, u in rows:
        products = await db.scalar(select(func.count()).select_from(Product).where(Product.store_id == s.id, Product.status != "archived"))
        channels = (await db.execute(select(Channel.platform).where(Channel.store_id == s.id, Channel.status != "disconnected"))).scalars().all()
        last = await db.scalar(select(func.max(Operation.created_at)).where(Operation.store_id == s.id))
        out.append({**StoreOut.model_validate(s).model_dump(), "owner": {"id": u.id, "phone": u.phone, "name": u.name},
                    "products": products, "platforms": channels, "last_activity": last})
    return out


@router.get("/stores/{store_id}/products")
async def store_products(store_id: int, admin: AdminUser, db: DB):
    rows = (await db.execute(select(Product).where(Product.store_id == store_id).order_by(Product.id.desc()).limit(200))).scalars()
    return [{"id": p.id, "title": p.title, "price": p.price, "stock_qty": p.stock_qty, "source": p.source,
             "status": p.status, "created_at": p.created_at} for p in rows]


@router.get("/channels")
async def all_channels(admin: AdminUser, db: DB, platform: str = "", only_errors: bool = False):
    q = select(Channel, Store).join(Store, Store.id == Channel.store_id).where(Channel.status != "disconnected")
    if platform:
        q = q.where(Channel.platform == platform)
    if only_errors:
        q = q.where(Channel.status.in_(["error", "expired"]))
    rows = (await db.execute(q.order_by(Channel.id.desc()))).all()
    return [{**ChannelOut.model_validate(c).model_dump(), "store": {"id": s.id, "name": s.name}} for c, s in rows]


# ---------- plans ----------

@router.get("/plans", response_model=list[PlanOut])
async def list_plans(admin: AdminUser, db: DB):
    return (await db.execute(select(Plan).order_by(Plan.sort_order, Plan.duration_months))).scalars().all()


@router.get("/plan-tiers", response_model=list[PlanTier])
async def get_tiers(admin: AdminUser, db: DB):
    return await all_tiers(db)


def _clean_limits(limits: dict) -> dict:
    out: dict[str, Any] = {k: max(0, int(limits.get(k) or 0)) for k in LIMIT_KEYS}
    out.update({k: bool(limits.get(k)) for k in FLAG_KEYS})
    return out


@router.put("/plan-tiers/{tier}", response_model=PlanTier)
async def put_tier(tier: str, body: PlanTier, admin: AdminUser, db: DB):
    """Creates or edits one plan: its name, features and limits, and a price per billing period."""
    tier = tier.strip()[:40]
    if not tier or not body.name.strip():
        raise HTTPException(422, "نام پلن لازم است.")
    if sorted(p.duration_months for p in body.prices) != list(DURATIONS):
        raise HTTPException(422, "قیمت هر سه مدت ۱، ۳ و ۱۲ ماهه لازم است.")
    if any(p.price_toman < 0 or p.compare_at_toman < 0 for p in body.prices):
        raise HTTPException(422, "قیمت نمی‌تواند منفی باشد.")
    if body.is_featured:  # one highlighted plan on the pricing page
        for p in (await db.execute(select(Plan).where(Plan.tier != tier, Plan.is_featured.is_(True)))).scalars():
            p.is_featured = False
    current = {p.duration_months: p for p in (await db.execute(select(Plan).where(Plan.tier == tier))).scalars()}
    features = [f.strip() for f in body.features if f.strip()]
    for price in body.prices:
        p = current.get(price.duration_months) or Plan(tier=tier, duration_months=price.duration_months)
        limits = _clean_limits(body.limits)
        if price.max_stores is not None:  # e.g. the yearly plan allows a second store
            limits["max_stores"] = max(0, price.max_stores)
        p.name, p.description, p.features, p.limits = body.name.strip(), body.description.strip(), features, limits
        p.price_toman, p.compare_at_toman, p.is_active = price.price_toman, price.compare_at_toman, price.is_active
        p.sort_order, p.is_featured = body.sort_order, body.is_featured
        db.add(p)
    audit(db, admin, "update_plan", "plan", None, {"tier": tier, **body.model_dump(exclude={"subscribers"})})
    await db.commit()
    return await tier_config(db, tier)


class TierActiveIn(BaseModel):
    is_active: bool


@router.post("/plan-tiers/{tier}/active", response_model=PlanTier)
async def set_tier_active(tier: str, body: TierActiveIn, admin: AdminUser, db: DB):
    """Stops or resumes selling a plan (all its periods). People already on it keep it until it ends."""
    rows = (await db.execute(select(Plan).where(Plan.tier == tier))).scalars().all()
    if not rows:
        raise HTTPException(404, "پلن پیدا نشد.")
    for p in rows:
        p.is_active = body.is_active
    audit(db, admin, "set_plan_active", "plan", None, {"tier": tier, "is_active": body.is_active})
    await db.commit()
    return await tier_config(db, tier)


@router.delete("/plan-tiers/{tier}")
async def delete_tier(tier: str, admin: AdminUser, db: DB):
    rows = (await db.execute(select(Plan).where(Plan.tier == tier))).scalars().all()
    if not rows:
        raise HTTPException(404, "پلن پیدا نشد.")
    ids = [p.id for p in rows]
    used = any([await db.scalar(select(m.id).where(m.plan_id.in_(ids)).limit(1)) for m in (Subscription, PlanRequest, Payment)])
    if used:
        raise HTTPException(409, "این پلن خریدار یا سابقه دارد و حذف نمی‌شود؛ به‌جایش فروشش را خاموش کنید.")
    for p in rows:
        await db.delete(p)
    audit(db, admin, "delete_plan", "plan", None, {"tier": tier})
    await db.commit()
    return {"ok": True}


class FreeLimitsIn(BaseModel):
    limits: dict[str, Any]


@router.get("/free-limits")
async def get_free_limits(admin: AdminUser, db: DB):
    return {"limits": await free_limits(db)}


@router.put("/free-limits")
async def put_free_limits(body: FreeLimitsIn, admin: AdminUser, db: DB):
    """Limits for sellers who never bought a plan (trial). After a plan ends no AI runs, whatever these say."""
    limits = await save_free_limits(db, _clean_limits(body.limits))
    audit(db, admin, "update_free_limits", "plan", None, limits)
    await db.commit()
    return {"limits": limits}


class InstagramAppIn(BaseModel):
    enabled: bool = False
    app_id: str = ""
    app_secret: str | None = None  # None keeps the stored secret
    proxy: str = ""


@router.get("/integrations/instagram")
async def get_instagram(admin: AdminUser, db: DB):
    app = await instagram_app(db)
    return {**{k: app[k] for k in ("enabled", "app_id", "proxy", "oauth_ready", "redirect_uri")},
            "has_secret": bool(app["app_secret"]), "scopes": INSTAGRAM_SCOPES}


@router.put("/integrations/instagram")
async def put_instagram(body: InstagramAppIn, admin: AdminUser, db: DB):
    await save_instagram_app(db, enabled=body.enabled, app_id=body.app_id, proxy=body.proxy, app_secret=body.app_secret)
    audit(db, admin, "update_integration", "instagram", None,
          {"enabled": body.enabled, "app_id": body.app_id, "proxy": bool(body.proxy), "secret_changed": body.app_secret is not None})
    await db.commit()
    return await get_instagram(admin, db)


class InstagramCrawlerIn(BaseModel):
    provider: str = "apify"
    token: str | None = None  # None keeps the stored token
    proxy: str = ""


@router.get("/integrations/instagram-crawler")
async def get_instagram_crawler(admin: AdminUser, db: DB):
    cfg = await instagram_crawler(db)
    token = cfg["token"]
    return {"provider": cfg["provider"], "proxy": cfg["proxy"], "has_token": bool(token),
            "token_hint": f"…{token[-4:]}" if len(token) > 8 else "", "providers": [{"id": "apify", "label": "Apify"}]}


@router.put("/integrations/instagram-crawler")
async def put_instagram_crawler(body: InstagramCrawlerIn, admin: AdminUser, db: DB):
    if body.provider != "apify":
        raise HTTPException(422, "فعلاً فقط Apify پشتیبانی می‌شود.")
    await save_instagram_crawler(db, provider=body.provider, proxy=body.proxy, token=body.token)
    audit(db, admin, "update_integration", "instagram_crawler", None,
          {"provider": body.provider, "proxy": bool(body.proxy), "token_changed": body.token is not None})
    await db.commit()
    return await get_instagram_crawler(admin, db)


@router.post("/integrations/instagram-crawler/test")
async def test_instagram_crawler(admin: AdminUser, db: DB):
    from app.services.instagram_crawl import check_key

    ok, message = await check_key(await instagram_crawler(db))
    return {"ok": ok, "message": message}


class PaymentConfigIn(BaseModel):
    enabled: bool = False
    bank: str = "ZARINPAL"
    gateways: dict[str, dict[str, Any]] = {}


@router.get("/integrations/payment")
async def get_payment(admin: AdminUser, db: DB):
    cfg = await payment_config(db)
    # Secrets are write-only: the panel sees which fields are set and only their last 4 characters.
    masked = {code: {f: (f"••••{str(v)[-4:]}" if v and f != "SANDBOX" else v) for f, v in vals.items()}
              for code, vals in cfg["gateways"].items()}
    return {"enabled": cfg["enabled"], "bank": cfg["bank"], "gateways": masked, "banks": BANKS,
            "callback_url": f"{settings.public_base_url.rstrip('/')}/api/v1/payments/callback/"}


@router.put("/integrations/payment")
async def put_payment(body: PaymentConfigIn, admin: AdminUser, db: DB):
    if body.bank not in BANKS:
        raise HTTPException(422, "درگاه نامعتبر است.")
    current = (await payment_config(db))["gateways"]
    merged: dict[str, dict[str, Any]] = {}
    for code in BANKS:
        old, new = current.get(code, {}), body.gateways.get(code, {})
        vals = {f: (old.get(f, "") if str(new.get(f, "")).startswith("••••") or f not in new else str(new[f]).strip())
                for f in BANKS[code]["fields"]}
        if BANKS[code].get("sandbox"):
            vals["SANDBOX"] = bool(new.get("SANDBOX", old.get("SANDBOX", False)))
        if any(vals.get(f) for f in BANKS[code]["fields"]):
            merged[code] = vals
    if body.enabled and not merged.get(body.bank):
        raise HTTPException(422, "اطلاعات درگاه انتخاب‌شده را کامل کنید.")
    await save_payment_config(db, enabled=body.enabled, bank=body.bank, gateways=merged)
    audit(db, admin, "update_integration", "payment", None, {"enabled": body.enabled, "bank": body.bank})
    await db.commit()
    return await get_payment(admin, db)


@router.get("/payments")
async def payments(admin: AdminUser, db: DB):
    rows = (await db.execute(select(Payment).options(selectinload(Payment.user), selectinload(Payment.plan))
                             .order_by(Payment.id.desc()).limit(200))).scalars()
    return [{"id": p.id, "user": {"id": p.user.id, "phone": p.user.phone, "name": p.user.name},
             "plan": f"{p.plan.name} {p.plan.duration_months} ماهه", "amount_toman": p.amount_toman, "bank_type": p.bank_type,
             "status": p.status, "tracking_code": p.tracking_code, "message": p.message, "created_at": p.created_at,
             "paid_at": p.paid_at} for p in rows]


@router.get("/plan-requests")
async def plan_requests(admin: AdminUser, db: DB, status: str = "pending"):
    rows = (await db.execute(select(PlanRequest, User, Plan).join(User, User.id == PlanRequest.user_id)
                             .join(Plan, Plan.id == PlanRequest.plan_id).where(PlanRequest.status == status)
                             .order_by(PlanRequest.id.desc()))).all()
    return [{"id": r.id, "status": r.status, "note": r.note, "created_at": r.created_at,
             "user": {"id": u.id, "phone": u.phone, "name": u.name},
             "plan": {"id": p.id, "name": p.name, "duration_months": p.duration_months, "price_toman": p.price_toman}}
            for r, u, p in rows]


class DecideIn(BaseModel):
    approve: bool


@router.post("/plan-requests/{req_id}")
async def decide_request(req_id: int, body: DecideIn, admin: AdminUser, db: DB):
    r = await db.get(PlanRequest, req_id)
    if not r or r.status != "pending":
        raise HTTPException(404, "درخواست پیدا نشد.")
    r.status = "approved" if body.approve else "rejected"
    if body.approve:
        await _activate(db, admin, r.user_id, r.plan_id, None, f"درخواست #{r.id}")
    await db.commit()
    return {"ok": True}


@router.get("/audit")
async def audit_log(admin: AdminUser, db: DB):
    rows = (await db.execute(select(AdminAuditLog, User).join(User, User.id == AdminAuditLog.admin_id)
                             .order_by(AdminAuditLog.id.desc()).limit(200))).all()
    return [{"id": a.id, "admin": u.phone, "action": a.action, "target_type": a.target_type, "target_id": a.target_id,
             "diff": a.diff, "created_at": a.created_at} for a, u in rows]


# ---------- AI keys, routing, monitoring ----------

class ProviderIn(BaseModel):
    name: str
    kind: str
    base_url: str = ""
    api_key: str = ""  # empty on update = keep the stored key
    proxy: str = ""  # optional http(s)/socks proxy for this key
    default_model: str = ""
    input_price_per_m: int = 0
    output_price_per_m: int = 0
    monthly_budget_toman: int = 0
    is_active: bool = True


def _provider_out(p: AiProvider) -> dict[str, Any]:
    return {"proxy": provider_proxy(p), **{c: getattr(p, c) for c in ("id", "name", "kind", "base_url", "key_hint", "default_model", "input_price_per_m",
                                         "output_price_per_m", "monthly_budget_toman", "is_active", "last_test_at",
                                         "last_test_ok", "last_test_message", "created_at")}}


@router.get("/ai/providers")
async def providers(admin: AdminUser, db: DB):
    return {"providers": [_provider_out(p) for p in (await db.execute(select(AiProvider).order_by(AiProvider.id))).scalars()],
            "kinds": [{"kind": "gemini", "label": "Google Gemini (کلید مستقیم AI Studio)", "base_url": DEFAULT_BASE_URLS["gemini"],
                       "hint": "کلید را از aistudio.google.com/apikey بگیر. گوگل ایران را تحریم کرده؛ روی سرور داخل ایران پروکسی لازم است.",
                       "model": "gemini-3.6-flash"},
                      {"kind": "metis", "label": "متیس", "base_url": DEFAULT_BASE_URLS["metis"],
                       "hint": "کلید متیس؛ از داخل ایران بدون پروکسی کار می‌کند. «دریافت مدل‌ها» همهٔ مدل‌های متیس (Gemini، GPT، Claude و DeepSeek) را می‌آورد.",
                       "model": "gemini-3.6-flash"},
                      {"kind": "avalai", "label": "اول‌ای‌آی (AvalAI)", "base_url": DEFAULT_BASE_URLS["avalai"],
                       "hint": "کلید را از chat.avalai.ir/platform بگیر؛ از داخل ایران بدون پروکسی کار می‌کند. «دریافت مدل‌ها» همهٔ مدل‌های اول‌ای‌آی (Gemini، GPT، Claude، DeepSeek و …) را می‌آورد.",
                       "model": "gemini-3.6-flash"},
                      {"kind": "openai", "label": "OpenAI یا سرویس سازگار", "base_url": DEFAULT_BASE_URLS["openai"],
                       "hint": "هر سرویس سازگار با OpenAI (مثلاً واسط‌های ایرانی)", "model": "gpt-4.1-mini"},
                      {"kind": "openrouter", "label": "OpenRouter", "base_url": DEFAULT_BASE_URLS["openrouter"],
                       "hint": "کلید را از openrouter.ai/keys بگیر. برای ساخت عکس: google/gemini-3.1-flash-lite-image (Nano Banana 2 Lite).",
                       "model": "google/gemini-3.6-flash"},
                      {"kind": "anthropic", "label": "Claude (Anthropic)", "base_url": DEFAULT_BASE_URLS["anthropic"], "hint": "", "model": "claude-sonnet-5-5"}],
            "tasks": AI_TASKS}


@router.post("/ai/providers")
async def add_provider(body: ProviderIn, admin: AdminUser, db: DB):
    if problem := check_model("product_generate", body.default_model.strip()):
        raise HTTPException(422, f"مدل پیش‌فرض: {problem}")
    if body.kind not in DEFAULT_BASE_URLS:
        raise HTTPException(422, "نوع سرویس نامعتبر است.")
    if not body.api_key.strip():
        raise HTTPException(422, "کلید API را وارد کنید.")
    p = AiProvider(**body.model_dump(exclude={"api_key", "proxy"}),
                   api_key_encrypted=encrypt_credentials({"api_key": body.api_key.strip(), "proxy": body.proxy.strip()}),
                   key_hint=body.api_key.strip()[-4:])
    db.add(p)
    await db.flush()
    audit(db, admin, "add_ai_key", "ai_provider", p.id, {"name": p.name, "kind": p.kind})
    await db.commit()
    return _provider_out(p)


@router.put("/ai/providers/{pid}")
async def update_provider(pid: int, body: ProviderIn, admin: AdminUser, db: DB):
    if problem := check_model("product_generate", body.default_model.strip()):
        raise HTTPException(422, f"مدل پیش‌فرض: {problem}")
    p = await db.get(AiProvider, pid)
    if not p:
        raise HTTPException(404, "کلید پیدا نشد.")
    if body.kind not in DEFAULT_BASE_URLS:
        raise HTTPException(422, "نوع سرویس نامعتبر است.")
    for k, v in body.model_dump(exclude={"api_key", "proxy"}).items():
        setattr(p, k, v)
    creds = decrypt_credentials(p.api_key_encrypted)
    if body.api_key.strip():
        creds["api_key"] = body.api_key.strip()
        p.key_hint = body.api_key.strip()[-4:]
    creds["proxy"] = body.proxy.strip()
    p.api_key_encrypted = encrypt_credentials(creds)
    p.updated_at = utcnow()
    audit(db, admin, "update_ai_key", "ai_provider", p.id, {"name": p.name, "key_changed": bool(body.api_key.strip())})
    await db.commit()
    return _provider_out(p)


@router.delete("/ai/providers/{pid}")
async def delete_provider(pid: int, admin: AdminUser, db: DB):
    p = await db.get(AiProvider, pid)
    if not p:
        raise HTTPException(404, "کلید پیدا نشد.")
    p.is_active = False
    p.api_key_encrypted = encrypt_credentials({})
    audit(db, admin, "delete_ai_key", "ai_provider", p.id)
    await db.commit()
    return {"ok": True}


class ModelsIn(BaseModel):
    kind: str
    base_url: str = ""
    api_key: str = ""
    proxy: str = ""


@router.post("/ai/models")
async def available_models(body: ModelsIn, admin: AdminUser, db: DB, provider_id: int | None = None):
    """Lists models for a key being typed in the form, or for a stored key (provider_id) when api_key is empty."""
    if body.kind not in DEFAULT_BASE_URLS:
        raise HTTPException(422, "نوع سرویس نامعتبر است.")
    stored = await db.get(AiProvider, provider_id) if provider_id else None
    creds = decrypt_credentials(stored.api_key_encrypted) if stored else {}
    key = body.api_key.strip() or creds.get("api_key", "")
    if not key:
        raise HTTPException(422, "اول کلید API را وارد کنید.")
    probe = AiProvider(name="probe", kind=body.kind, base_url=body.base_url,
                       api_key_encrypted=encrypt_credentials({"api_key": key, "proxy": body.proxy.strip()}))
    try:
        ids = await list_models(probe)
        return {"models": [m for m in ids if model_kind(m)[0] == "text"],
                "image_models": [m for m in ids if model_kind(m)[0] == "image"]}
    except AIError as exc:
        raise HTTPException(400, exc.message_fa)
    except httpx.HTTPError:
        raise HTTPException(400, "ارتباط با سرویس برقرار نشد؛ پروکسی یا VPN را بررسی کنید.")


@router.post("/ai/providers/{pid}/test")
async def test_key(pid: int, admin: AdminUser, db: DB):
    p = await db.get(AiProvider, pid)
    if not p:
        raise HTTPException(404, "کلید پیدا نشد.")
    ok, msg, ms = await test_provider(p)
    p.last_test_at, p.last_test_ok, p.last_test_message = utcnow(), ok, msg
    await db.commit()
    return {"ok": ok, "message": msg, "latency_ms": ms}


@router.get("/ai/routes")
async def routes(admin: AdminUser, db: DB):
    existing = {r.task: r for r in (await db.execute(select(AiTaskRoute))).scalars()}
    return [{"task": t, "label": label, "needs": "image" if t in IMAGE_TASKS else "text",
             **({c: getattr(existing[t], c) for c in ("provider_id", "model", "fallback_provider_id", "fallback_model", "temperature")}
                if t in existing else {"provider_id": None, "model": "", "fallback_provider_id": None, "fallback_model": "", "temperature": 30})}
            for t, label in AI_TASKS.items()]


class RouteIn(BaseModel):
    provider_id: int | None = None
    model: str = ""
    fallback_provider_id: int | None = None
    fallback_model: str = ""
    temperature: int = 30


@router.put("/ai/routes/{task}")
async def set_route(task: str, body: RouteIn, admin: AdminUser, db: DB):
    if task not in AI_TASKS:
        raise HTTPException(404, "کار نامعتبر است.")
    for model, pid, label in ((body.model, body.provider_id, "مدل اصلی"), (body.fallback_model, body.fallback_provider_id, "مدل جایگزین")):
        if not pid:
            continue
        p = await db.get(AiProvider, pid)
        effective = model.strip() or (p.default_model if p and task not in IMAGE_TASKS else "")
        if problem := check_model(task, effective):
            raise HTTPException(422, f"{label}: {problem}")
        if task in IMAGE_TASKS and p and p.kind not in IMAGE_KINDS:
            raise HTTPException(422, f"{label}: ساخت عکس فقط با کلید Gemini (مستقیم، متیس یا اول‌ای‌آی) یا OpenRouter کار می‌کند.")
        if task in IMAGE_TASKS and p and p.kind in ("metis", "avalai") and metis_family(effective) != "gemini":
            raise HTTPException(422, f"{label}: ساخت عکس با {'متیس' if p.kind == 'metis' else 'اول‌ای‌آی'} فقط با مدل‌های تصویری Gemini کار می‌کند (مثل gemini-3.1-flash-image).")
    r = (await db.execute(select(AiTaskRoute).where(AiTaskRoute.task == task))).scalar_one_or_none()
    if not r:
        r = AiTaskRoute(task=task)
        db.add(r)
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    r.updated_at = utcnow()  # a re-save re-enables an image route that was paused for quota
    audit(db, admin, "set_ai_route", "ai_route", None, {"task": task, **body.model_dump()})
    await db.commit()
    return {"ok": True}


@router.get("/ai/usage")
async def usage(admin: AdminUser, db: DB, days: int = 30, provider_id: int | None = None):
    since = utcnow() - timedelta(days=days)
    base = [AiUsageLog.created_at >= since]
    if provider_id:
        base.append(AiUsageLog.provider_id == provider_id)
    agg = lambda: (func.count(), func.sum(case((AiUsageLog.ok, 1), else_=0)), func.coalesce(func.sum(AiUsageLog.input_tokens), 0),  # noqa: E731
                   func.coalesce(func.sum(AiUsageLog.output_tokens), 0), func.coalesce(func.sum(AiUsageLog.cost_toman), 0),
                   func.coalesce(func.avg(AiUsageLog.latency_ms), 0))

    def row(r) -> dict:
        calls, ok, tin, tout, cost, lat = r
        return {"calls": calls, "ok": int(ok or 0), "failed": calls - int(ok or 0),
                "error_rate": round(100 * (calls - int(ok or 0)) / calls, 1) if calls else 0,
                "input_tokens": int(tin), "output_tokens": int(tout), "cost_toman": int(cost), "avg_latency_ms": int(lat)}

    month_start = datetime.combine(utcnow().date().replace(day=1), time.min, tzinfo=timezone.utc)
    per_provider = []
    for p in (await db.execute(select(AiProvider).order_by(AiProvider.id))).scalars():
        r = (await db.execute(select(*agg()).where(*base, AiUsageLog.provider_id == p.id))).one()
        month_cost = await db.scalar(select(func.coalesce(func.sum(AiUsageLog.cost_toman), 0)).where(
            AiUsageLog.provider_id == p.id, AiUsageLog.created_at >= month_start))
        last_used = await db.scalar(select(func.max(AiUsageLog.created_at)).where(AiUsageLog.provider_id == p.id))
        per_provider.append({"provider": _provider_out(p), **row(r), "month_cost_toman": int(month_cost), "last_used": last_used,
                             "budget_used_pct": round(100 * month_cost / p.monthly_budget_toman, 1) if p.monthly_budget_toman else None})
    day_col = func.date(AiUsageLog.created_at)
    daily = [{"day": str(d), **row((c, ok, tin, tout, cost, lat))} for d, c, ok, tin, tout, cost, lat in
             (await db.execute(select(day_col, *agg()).where(*base).group_by(day_col).order_by(day_col))).all()]
    by_task = [{"task": t, "label": AI_TASKS.get(t, "تست کلید" if t == "key_test" else t), **row((c, ok, tin, tout, cost, lat))}
               for t, c, ok, tin, tout, cost, lat in
               (await db.execute(select(AiUsageLog.task, *agg()).where(*base).group_by(AiUsageLog.task))).all()]
    by_store = []
    for sid, c, ok, tin, tout, cost, lat in (await db.execute(select(AiUsageLog.store_id, *agg()).where(*base, AiUsageLog.store_id.is_not(None))
                                                              .group_by(AiUsageLog.store_id).order_by(func.count().desc()).limit(20))).all():
        s = await db.get(Store, sid)
        by_store.append({"store_id": sid, "store": s.name if s else "?", **row((c, ok, tin, tout, cost, lat))})
    errors = [{"id": e.id, "created_at": e.created_at, "provider_id": e.provider_id, "task": e.task, "model": e.model,
               "error_code": e.error_code, "error_message": e.error_message}
              for e in (await db.execute(select(AiUsageLog).where(*base, ~AiUsageLog.ok).order_by(AiUsageLog.id.desc()).limit(30))).scalars()]
    totals = row((await db.execute(select(*agg()).where(*base))).one())
    return {"totals": totals, "providers": per_provider, "daily": daily, "by_task": by_task, "by_store": by_store, "errors": errors}
