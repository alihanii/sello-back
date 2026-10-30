"""Calendar and dashboard: both aggregate activity_events and operations."""

from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter
from sqlalchemy import func, select

from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.models import ActivityEvent, Approval, Automation, Channel, ChannelListing, CrawlSource, Operation, Plan, Product
from app.schemas import OperationOut
from app.services.ai import ai_available, route_problem
from app.services.plans import active_subscription, limits_for

router = APIRouter(tags=["insights"])


@router.get("/ai/status")
async def ai_status(user: CurrentUser):
    """Whether any AI key is configured, so the UI can explain why AI features are off."""
    image_problem = await route_problem("image_edit")
    # The technical reason (keys, quota, billing) is for the admin only; sellers just don't see the feature.
    return {"available": await ai_available("product_generate") or await ai_available("caption_rewrite"),
            "image_edit": not image_problem, "image_edit_problem": image_problem if user.role == "admin" else ""}
EVENT_TYPES = ["crawl_run", "crawl_item", "crawl_failed", "product_created", "publish_ok", "publish_failed"]


async def _daily(db, store_id: int, start: date, end: date) -> dict[str, dict[str, int]]:
    rows = (await db.execute(
        select(ActivityEvent.day, ActivityEvent.type, func.sum(ActivityEvent.count))
        .where(ActivityEvent.store_id == store_id, ActivityEvent.day >= start, ActivityEvent.day <= end)
        .group_by(ActivityEvent.day, ActivityEvent.type))).all()
    out: dict[str, dict[str, int]] = {}
    for day, typ, n in rows:
        out.setdefault(day.isoformat(), {t: 0 for t in EVENT_TYPES})[typ] = int(n)
    return out


@router.get("/stores/{store_id}/calendar")
async def calendar(store_id: int, user: CurrentUser, db: DB, start: date, end: date):
    await get_store_for(db, user, store_id)
    days = await _daily(db, store_id, start, end)
    # Days where a scheduled crawl source existed but no crawl ran: flagged on the calendar.
    sources = (await db.execute(select(CrawlSource).where(CrawlSource.store_id == store_id, CrawlSource.is_active,
                                                          CrawlSource.schedule != "manual"))).scalars().all()
    expected_from = min((s.created_at.date() for s in sources), default=None)
    today = utcnow().date()
    flags: dict[str, list[str]] = {}
    d = start
    while d <= min(end, today):
        k = d.isoformat()
        stats = days.get(k, {})
        if expected_from and d >= expected_from and not stats.get("crawl_run"):
            flags.setdefault(k, []).append("no_crawl")
        elif stats.get("crawl_run") and not stats.get("crawl_item"):
            flags.setdefault(k, []).append("no_data")
        if stats.get("publish_failed") or stats.get("crawl_failed"):
            flags.setdefault(k, []).append("errors")
        d += timedelta(days=1)
    upcoming = []
    for s in sources:
        if s.next_run_at:
            upcoming.append({"type": "crawl", "title_fa": f"کرال «{s.name}»", "at": s.next_run_at, "schedule": s.schedule})
    rules = (await db.execute(select(Automation).where(Automation.store_id == store_id, Automation.trigger == "schedule",
                                                       Automation.is_active))).scalars()
    for r in rules:
        upcoming.append({"type": "publish", "title_fa": f"انتشار زمان‌بندی‌شده «{r.name}»",
                         "hour": (r.actions or {}).get("schedule_hour", 10), "schedule": "daily"})
    # 7-day average of incoming data for "today vs average".
    week = await _daily(db, store_id, today - timedelta(days=7), today - timedelta(days=1))
    avg_items = sum(v.get("crawl_item", 0) for v in week.values()) / 7
    return {"days": days, "flags": flags, "upcoming": upcoming, "avg_items_7d": round(avg_items, 1)}


@router.get("/stores/{store_id}/calendar/day/{day}")
async def calendar_day(store_id: int, day: date, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    ops = (await db.execute(select(Operation).where(Operation.store_id == store_id, Operation.created_at >= start,
                                                    Operation.created_at < start + timedelta(days=1))
                            .order_by(Operation.created_at))).scalars().all()
    stats = (await _daily(db, store_id, day, day)).get(day.isoformat(), {t: 0 for t in EVENT_TYPES})
    return {"day": day, "stats": stats, "operations": [OperationOut.model_validate(o) for o in ops]}


@router.get("/stores/{store_id}/dashboard")
async def dashboard(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    today = utcnow().date()
    products = await db.scalar(select(func.count()).select_from(Product).where(
        Product.store_id == store_id, Product.status != "archived"))
    published_any = await db.scalar(select(func.count(func.distinct(ChannelListing.product_id))).join(Product).where(
        Product.store_id == store_id, ChannelListing.status == "published"))
    series = await _daily(db, store_id, today - timedelta(days=29), today)
    t = series.get(today.isoformat(), {})
    ok30 = sum(v.get("publish_ok", 0) for v in series.values())
    fail30 = sum(v.get("publish_failed", 0) for v in series.values())
    attention = []
    for ch in (await db.execute(select(Channel).where(Channel.store_id == store_id,
                                                      Channel.status.in_(["error", "expired"])))).scalars():
        attention.append({"type": "channel", "id": ch.id, "text": f"کانال «{ch.display_name or ch.platform}» قطع است: {ch.last_error}"})
    failed = await db.scalar(select(func.count()).select_from(Operation).where(
        Operation.store_id == store_id, Operation.status == "failed", ~Operation.seen))
    if failed:
        attention.append({"type": "operations", "text": f"{failed} عملیات ناموفق دیده‌نشده"})
    pending = await db.scalar(select(func.count()).select_from(Approval).where(
        Approval.store_id == store_id, Approval.status == "pending"))
    if pending:
        attention.append({"type": "approvals", "text": f"{pending} محصول منتظر تأیید انتشار"})
    out_of_stock = await db.scalar(select(func.count()).select_from(Product).where(
        Product.store_id == store_id, Product.status == "ready", Product.stock_qty <= 0))
    if out_of_stock:
        attention.append({"type": "stock", "text": f"موجودی {out_of_stock} محصول تمام شده"})
    drafts = await db.scalar(select(func.count()).select_from(Product).where(
        Product.store_id == store_id, Product.status == "draft"))
    if drafts:
        attention.append({"type": "inbox", "text": f"{drafts} پیش‌نویس (کرال یا AI) منتظر بررسی"})
    month_start = datetime.combine(today.replace(day=1), time.min, tzinfo=timezone.utc)
    publishes_month = await db.scalar(select(func.count()).select_from(Operation).where(
        Operation.store_id == store_id, Operation.type == "publish", Operation.status == "succeeded",
        Operation.created_at >= month_start))
    limits = await limits_for(db, user)
    sub = await active_subscription(db, user.id)
    plan = await db.get(Plan, sub.plan_id) if sub else None
    return {
        "kpis": {"products": products, "published_any": published_any, "unpublished": products - published_any,
                 "published_today": t.get("publish_ok", 0), "crawled_today": t.get("crawl_item", 0),
                 "success_rate": round(100 * ok30 / (ok30 + fail30)) if ok30 + fail30 else None},
        "series": series,
        "attention": attention,
        "usage": {"publishes_month": publishes_month, "limits": limits,
                  "plan": plan.name if plan else None, "ends_at": sub.ends_at if sub else None},
    }
