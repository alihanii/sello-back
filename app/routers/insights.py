"""Calendar and dashboard: both aggregate activity_events and operations."""

from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter
from sqlalchemy import case, func, select

from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.models import ActivityEvent, AiOrder, Channel, ChatConversation, ChatMessage, ChannelListing, CrawlSource, Operation, Plan, Product
from app.schemas import OperationOut
from app.services.ai import ai_available, route_problem
from app.services.plans import active_subscription, limits_for

router = APIRouter(tags=["insights"])


@router.get("/ai/status")
async def ai_status(user: CurrentUser):
    """Whether any AI key is configured, so the UI can explain why AI features are off."""
    image_problem = await route_problem("image_edit")
    # The technical reason (keys, quota, billing) is for the admin only; sellers just don't see the feature.
    return {"available": await ai_available("product_generate"),
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


CHAT_TYPES = ["orders", "orders_toman", "chats_answered", "ai_replies"]


def _day_key(v) -> str:
    return v.isoformat() if hasattr(v, "isoformat") else str(v)[:10]


async def _chat_daily(db, store_id: int, start: date, end: date) -> dict[str, dict[str, int]]:
    """Per day: orders the assistant took and chats it answered (test chats left out)."""
    lo = datetime.combine(start, time.min, tzinfo=timezone.utc)
    hi = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
    out: dict[str, dict[str, int]] = {}
    oday = func.date(AiOrder.created_at)
    live = AiOrder.status.notin_(["cancelled", "rejected"])
    for d, n, toman in (await db.execute(
            select(oday, func.count(), func.coalesce(func.sum(case((live, AiOrder.total_toman), else_=0)), 0))
            .where(AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.created_at >= lo, AiOrder.created_at < hi)
            .group_by(oday))).all():
        row = out.setdefault(_day_key(d), {t: 0 for t in CHAT_TYPES})
        row["orders"], row["orders_toman"] = int(n), int(toman or 0)
    mday = func.date(ChatMessage.created_at)
    for d, convs, msgs in (await db.execute(
            select(mday, func.count(func.distinct(ChatMessage.conversation_id)), func.count())
            .where(ChatMessage.store_id == store_id, ChatMessage.role == "ai", ~ChatMessage.is_test,
                   ChatMessage.created_at >= lo, ChatMessage.created_at < hi)
            .group_by(mday))).all():
        row = out.setdefault(_day_key(d), {t: 0 for t in CHAT_TYPES})
        row["chats_answered"], row["ai_replies"] = int(convs), int(msgs)
    return out


async def _all_daily(db, store_id: int, start: date, end: date) -> dict[str, dict[str, int]]:
    days = await _daily(db, store_id, start, end)
    for k, chat in (await _chat_daily(db, store_id, start, end)).items():
        days.setdefault(k, {t: 0 for t in EVENT_TYPES}).update(chat)
    for v in days.values():
        for t in CHAT_TYPES:
            v.setdefault(t, 0)
    return days


@router.get("/stores/{store_id}/calendar")
async def calendar(store_id: int, user: CurrentUser, db: DB, start: date, end: date):
    await get_store_for(db, user, store_id)
    days = await _all_daily(db, store_id, start, end)
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
    stats = (await _all_daily(db, store_id, day, day)).get(day.isoformat(), {t: 0 for t in EVENT_TYPES + CHAT_TYPES})
    end = start + timedelta(days=1)
    orders = (await db.execute(select(AiOrder).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.created_at >= start, AiOrder.created_at < end)
        .order_by(AiOrder.created_at))).scalars().all()
    replies = func.count(ChatMessage.id)
    chats = (await db.execute(
        select(ChatConversation.id, ChatConversation.customer_name, ChatConversation.customer_username,
               ChatConversation.platform, replies, func.max(ChatMessage.created_at))
        .join(ChatMessage, ChatMessage.conversation_id == ChatConversation.id)
        .where(ChatMessage.store_id == store_id, ChatMessage.role == "ai", ~ChatMessage.is_test,
               ChatMessage.created_at >= start, ChatMessage.created_at < end)
        .group_by(ChatConversation.id).order_by(func.max(ChatMessage.created_at).desc()).limit(100))).all()
    return {"day": day, "stats": stats, "operations": [OperationOut.model_validate(o) for o in ops],
            "orders": [{"id": o.id, "created_at": o.created_at, "customer_name": o.customer_name,
                        "total_toman": o.total_toman, "status": o.status, "tracking_code": o.tracking_code,
                        "conversation_id": o.conversation_id, "platform": o.platform} for o in orders],
            "chats": [{"id": cid, "customer_name": name or (f"@{uname}" if uname else ""), "platform": plat,
                       "ai_replies": int(n), "last_at": last} for cid, name, uname, plat, n, last in chats]}


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
    out_of_stock = await db.scalar(select(func.count()).select_from(Product).where(
        Product.store_id == store_id, Product.status == "ready", Product.stock_qty <= 0))
    if out_of_stock:
        attention.append({"type": "stock", "text": f"موجودی {out_of_stock} محصول تمام شده"})
    drafts = await db.scalar(select(func.count()).select_from(Product).where(
        Product.store_id == store_id, Product.status == "draft"))
    if drafts:
        attention.append({"type": "inbox", "text": f"{drafts} پیش‌نویس (کرال یا AI) منتظر بررسی"})
    receipts = await db.scalar(select(func.count()).select_from(AiOrder).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.status == "payment_submitted"))
    if receipts:
        attention.append({"type": "ai_orders", "text": f"{receipts} سفارش چت با فیش پرداخت منتظر تأیید"})
    needs_human = await db.scalar(select(func.count()).select_from(ChatConversation).where(
        ChatConversation.store_id == store_id, ~ChatConversation.is_test, ChatConversation.status == "needs_human"))
    if needs_human:
        attention.append({"type": "chat", "text": f"{needs_human} گفتگوی چت منتظر جواب ادمین"})
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
