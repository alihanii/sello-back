"""Plans: one or more plans (tiers), each sold for 1/3/12 months; their limits; and what a seller has used."""

from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from fastapi import HTTPException
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import utcnow
from app.models import AiUsageLog, AppSetting, Channel, ChatMessage, Plan, Product, Store, Subscription, User

PLAN_TIER = "sello"  # the default plan
DURATIONS = (1, 3, 12)
DISCOUNTS = {1: 0, 3: 10, 12: 25}  # % off the monthly price, for new tiers

LIMIT_KEYS = ("max_stores", "max_channels", "max_products", "monthly_publishes", "monthly_ai_credits",
              "monthly_chat_messages")

# Limits for users who never had a plan (trial). The admin edits them (app setting `plan:free`).
FREE_LIMITS = {"max_stores": 1, "max_channels": 2, "max_products": 30, "monthly_ai_credits": 30,
               "monthly_publishes": 50, "crawl_frequency": "daily",
               "monthly_chat_messages": 100}
FREE_KEY = "plan:free"
# Limits added after a plan was saved: a paid plan without the key gets this instead of the free value.
PAID_DEFAULTS = {"monthly_chat_messages": 5000}
# After a plan ends (or the account is blocked) the seller keeps their data, but no AI runs for them.
NO_AI = {"monthly_ai_credits": 0, "monthly_chat_messages": 0}

# AI tasks that are part of the sales chat; they count against monthly_chat_messages, not AI credits.
CHAT_TASKS = {"chat", "chat_vision", "chat_router"}
# Credits per successful call; every other non-chat task costs 1. A generated picture costs ~20× a text call.
CREDIT_WEIGHT = {"image_edit": 20}

# The plan created on first start (prices in toman per month); the admin edits everything afterwards and can add
# more plans. Worst case AI cost on Gemini 2.5 Flash-Lite: 2,000 chat replies × ~63 t + 200 credits × ~400 t ≈ 206k t.
DEFAULT_TIERS = [
    {"tier": PLAN_TIER, "name": "فروشیفای", "monthly": 790_000, "sort_order": 1, "is_featured": False,
     "description": "همهٔ امکانات؛ فقط مدت اشتراک فرق می‌کند.",
     "limits": {"max_stores": 2, "max_channels": 6, "max_products": 1500, "monthly_publishes": 2000,
                "monthly_ai_credits": 200, "crawl_frequency": "hourly", "monthly_chat_messages": 2000},
     "features": ["۲ فروشگاه", "۶ کانال (تلگرام، بله، دیجی‌کالا، اینستاگرام)", "۱٬۵۰۰ محصول", "۲٬۰۰۰ انتشار در ماه",
                  "کرال ساعتی", "۲۰۰ اعتبار AI در ماه (ساخت محصول، کرال، عکس)", "دستیار چت فروش: ۲٬۰۰۰ جواب AI در ماه"]},
]


def tier_plans(t: dict) -> list[Plan]:
    out = []
    for months in DURATIONS:
        full = t["monthly"] * months
        out.append(Plan(name=t["name"], tier=t["tier"], duration_months=months,
                        price_toman=int(round(full * (100 - DISCOUNTS[months]) / 100, -4)),
                        compare_at_toman=full if DISCOUNTS[months] else 0, description=t["description"],
                        features=list(t["features"]), limits=dict(t["limits"]), is_featured=t["is_featured"],
                        is_active=True, sort_order=t["sort_order"]))
    return out


async def create_subscription(db: AsyncSession, user_id: int, plan: Plan, *, activated_by: str, note: str = "",
                              starts_at: datetime | None = None, replace: bool = False) -> Subscription:
    """Renewals stack after the current period instead of overlapping it. `replace` ends the current
    subscriptions and starts the new plan now (an admin changing the plan)."""
    now = utcnow()
    if replace:
        for s in (await db.execute(select(Subscription).where(Subscription.user_id == user_id, Subscription.status == "active",
                                                              Subscription.ends_at > now))).scalars():
            s.status = "cancelled"
            s.note = ((s.note + " · ") if s.note else "") + "پلن عوض شد"
        current = None
    else:
        current = (await db.execute(select(Subscription).where(Subscription.user_id == user_id, Subscription.status == "active",
                                                               Subscription.ends_at > now).order_by(Subscription.ends_at.desc()))).scalars().first()
    start = starts_at or (current.ends_at.replace(tzinfo=timezone.utc) if current else now)
    s = Subscription(user_id=user_id, plan_id=plan.id, status="active", starts_at=start,
                     ends_at=start + relativedelta(months=plan.duration_months), activated_by=activated_by, note=note)
    db.add(s)
    return s


async def active_subscription(db: AsyncSession, user_id: int) -> Subscription | None:
    now = utcnow()
    q = (select(Subscription).where(Subscription.user_id == user_id, Subscription.status == "active",
                                    Subscription.starts_at <= now, Subscription.ends_at > now)
         .order_by(Subscription.ends_at.desc()))
    return (await db.execute(q)).scalars().first()


async def free_limits(db: AsyncSession) -> dict:
    row = await db.get(AppSetting, FREE_KEY)
    return {**FREE_LIMITS, **(row.value if row else {})}


async def save_free_limits(db: AsyncSession, limits: dict) -> dict:
    row = await db.get(AppSetting, FREE_KEY)
    clean = {k: v for k, v in limits.items() if k in FREE_LIMITS}
    if row:
        row.value = clean
    else:
        db.add(AppSetting(key=FREE_KEY, value=clean))
    return {**FREE_LIMITS, **clean}


async def plan_status(db: AsyncSession, user: User) -> str:
    """active | free (never had a plan) | expired | blocked | admin."""
    if user.role == "admin":
        return "admin"
    if not user.is_active:
        return "blocked"
    if await active_subscription(db, user.id):
        return "active"
    had = await db.scalar(select(Subscription.id).where(Subscription.user_id == user.id, Subscription.starts_at <= utcnow())
                          .limit(1))
    return "expired" if had else "free"


async def limits_for(db: AsyncSession, user: User) -> dict:
    if user.role == "admin":
        return {k: (10**9 if isinstance(v, int) else "hourly") for k, v in FREE_LIMITS.items()}
    status = await plan_status(db, user)
    if status == "active":
        sub = await active_subscription(db, user.id)
        plan = await db.get(Plan, sub.plan_id)
        return {**FREE_LIMITS, **PAID_DEFAULTS, **(plan.limits if plan else {})}
    base = await free_limits(db)
    return {**base, **NO_AI} if status in ("expired", "blocked") else base


def _month_start() -> datetime:
    now = utcnow()
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


async def _store_ids(db: AsyncSession, owner_id: int) -> list[int]:
    return list((await db.execute(select(Store.id).where(Store.owner_id == owner_id))).scalars())


async def chat_used(db: AsyncSession, owner_id: int) -> int:
    """AI chat replies this month across all of the owner's stores (test chats aren't counted)."""
    ids = await _store_ids(db, owner_id)
    if not ids:
        return 0
    return await db.scalar(select(func.count()).select_from(ChatMessage).where(
        ChatMessage.store_id.in_(ids), ChatMessage.role == "ai", ChatMessage.model != "", ~ChatMessage.is_test,
        ChatMessage.created_at >= _month_start())) or 0


async def credits_used(db: AsyncSession, owner_id: int) -> int:
    ids = await _store_ids(db, owner_id)
    if not ids:
        return 0
    weight = case(*((AiUsageLog.task == t, w) for t, w in CREDIT_WEIGHT.items()), else_=1)
    return int(await db.scalar(select(func.coalesce(func.sum(weight), 0)).where(
        AiUsageLog.store_id.in_(ids), AiUsageLog.ok, AiUsageLog.task.notin_(CHAT_TASKS),
        AiUsageLog.created_at >= _month_start())) or 0)


async def credit_problem(db: AsyncSession, store_id: int, task: str) -> str:
    """Why this store's owner can't spend AI credits on `task` now, in Persian; empty when they can."""
    if task in CHAT_TASKS:
        return ""
    store = await db.get(Store, store_id)
    owner = await db.get(User, store.owner_id) if store else None
    if not owner or owner.role == "admin":
        return ""
    status = await plan_status(db, owner)
    if status == "blocked":
        return "حساب کاربری غیرفعال است؛ کارهای هوش مصنوعی انجام نمی‌شود."
    if status == "expired":
        return "پلن شما تمام شده؛ برای استفاده از هوش مصنوعی پلن را تمدید کنید."
    limit = int((await limits_for(db, owner)).get("monthly_ai_credits", 0))
    if await credits_used(db, owner.id) + CREDIT_WEIGHT.get(task, 1) > limit:
        return f"اعتبار AI این ماه ({limit}) تمام شده؛ برای ادامه پلن را ارتقا دهید."
    return ""


def _limit_error(what: str, limit: int) -> HTTPException:
    return HTTPException(402, {"message": f"به سقف {what} پلن فعلی ({limit}) رسیده‌اید؛ برای ادامه پلن را ارتقا دهید.",
                               "action": {"type": "upgrade", "label": "ارتقای پلن"}})


async def check_limit(db: AsyncSession, user: User, kind: str) -> None:
    limits = await limits_for(db, user)
    if kind == "stores":
        n = await db.scalar(select(func.count()).select_from(Store).where(Store.owner_id == user.id, ~Store.archived))
        if n >= limits["max_stores"]:
            raise _limit_error("تعداد فروشگاه", limits["max_stores"])
    elif kind == "channels":
        n = await db.scalar(select(func.count()).select_from(Channel).join(Store).where(
            Store.owner_id == user.id, Channel.status != "disconnected"))
        if n >= limits["max_channels"]:
            raise _limit_error("تعداد کانال", limits["max_channels"])
    elif kind == "products":
        n = await db.scalar(select(func.count()).select_from(Product).join(Store).where(
            Store.owner_id == user.id, Product.status != "archived"))
        if n >= limits["max_products"]:
            raise _limit_error("تعداد محصول", limits["max_products"])


async def tier_config(db: AsyncSession, tier: str) -> dict | None:
    """One tier as one feature set plus a price per billing period."""
    rows = (await db.execute(select(Plan).where(Plan.tier == tier).order_by(Plan.duration_months))).scalars().all()
    if not rows:
        return None
    base = rows[0]
    by_months = {p.duration_months: p for p in rows}
    subscribers = await db.scalar(select(func.count(func.distinct(Subscription.user_id))).where(
        Subscription.plan_id.in_([p.id for p in rows]), Subscription.status == "active", Subscription.ends_at > utcnow())) or 0
    return {
        "tier": tier, "name": base.name, "description": base.description, "features": base.features,
        "limits": base.limits, "sort_order": base.sort_order, "is_featured": any(p.is_featured for p in rows),
        "is_active": any(p.is_active for p in rows), "subscribers": subscribers,
        "prices": [{"duration_months": m, "plan_id": by_months[m].id if m in by_months else None,
                    "price_toman": by_months[m].price_toman if m in by_months else 0,
                    "compare_at_toman": by_months[m].compare_at_toman if m in by_months else 0,
                    "is_active": by_months[m].is_active if m in by_months else False} for m in DURATIONS],
    }


async def all_tiers(db: AsyncSession) -> list[dict]:
    tiers = (await db.execute(select(Plan.tier, func.min(Plan.sort_order)).group_by(Plan.tier)
                              .order_by(func.min(Plan.sort_order), Plan.tier))).all()
    return [c for t, _ in tiers if (c := await tier_config(db, t))]
