"""Status center plumbing: every action is an Operation with ordered steps. Handlers run
inside an OpContext; `ctx.step()` times the step and stores its (redacted) request/response."""

import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import utcnow
from app.core.security import redact
from app.models import ActivityEvent, Channel, ChannelListing, Operation, OperationStep, Product
from app.platforms.base import PlatformError
from app.services.ai import AIError


async def enqueue(db: AsyncSession, *, store_id: int, type: str, title_fa: str, payload: dict[str, Any] | None = None,
                  channel_id: int | None = None, product_id: int | None = None, trigger: str = "user",
                  actor: str = "user", ref_type: str = "", ref_id: int | None = None) -> Operation:
    op = Operation(store_id=store_id, type=type, title_fa=title_fa, payload=payload or {}, channel_id=channel_id,
                   product_id=product_id, trigger=trigger, actor=actor, ref_type=ref_type, ref_id=ref_id,
                   status="queued")
    db.add(op)
    await db.flush()
    return op


async def record_event(db: AsyncSession, store_id: int, type: str, count: int = 1, operation_id: int | None = None,
                       channel_id: int | None = None, payload: dict | None = None, day: date | None = None) -> None:
    if count <= 0:
        return
    db.add(ActivityEvent(store_id=store_id, day=day or utcnow().date(), type=type, count=count,
                         operation_id=operation_id, channel_id=channel_id, payload=payload or {}))


class _Step:
    def __init__(self) -> None:
        self.response: Any = {}
        self.message_fa = ""
        self.request: Any = None


class OpContext:
    def __init__(self, db: AsyncSession, op: Operation):
        self.db = db
        self.op = op
        self._order = len(op.steps) if op.steps else 0
        self.empty_reason: str = ""  # set by a crawl adapter to explain why it found nothing

    @asynccontextmanager
    async def step(self, name: str, label_fa: str, request: Any = None):
        self._order += 1
        row = OperationStep(operation_id=self.op.id, order=self._order, name=name, label_fa=label_fa,
                            status="running", started_at=utcnow(), request=redact(request or {}))
        self.db.add(row)
        self.op.current_step = label_fa
        await self.db.commit()
        handle = _Step()
        started = time.monotonic()
        try:
            yield handle
        except (PlatformError, AIError) as exc:
            row.status = "failed"
            row.message_fa = exc.message_fa
            row.response = redact(exc.detail) if exc.detail is not None else {}
            raise
        except Exception as exc:
            row.status = "failed"
            row.message_fa = "خطای داخلی سیستم."
            row.response = {"error": repr(exc)[:500]}
            raise
        else:
            row.status = "succeeded"
            row.message_fa = handle.message_fa
            row.response = redact(handle.response)
            if handle.request is not None:
                row.request = redact(handle.request)
        finally:
            row.duration_ms = int((time.monotonic() - started) * 1000)
            await self.db.commit()


def fail_operation(op: Operation, exc: Exception) -> None:
    op.status = "failed"
    op.finished_at = utcnow()
    if isinstance(exc, (PlatformError, AIError)):
        op.error_code = exc.code
        op.error_message_fa = exc.message_fa
        op.suggested_action = getattr(exc, "suggested_action", {}) or (
            {"type": "admin_ai", "label": "تنظیم کلید AI توسط ادمین"} if exc.code == "ai_not_configured" else {}
        )
    else:
        op.error_code = "internal"
        op.error_message_fa = "خطای داخلی سیستم؛ تیم پشتیبانی مطلع شد."
        op.suggested_action = {"type": "retry", "label": "تلاش دوباره"}


FORMAT_FA = {"post": "", "story": "استوری ", "reel": "ریلز "}


def turn_text(local: datetime, today, app: dict[str, Any], paused: bool = False) -> str:
    """The queued step text (the column holds 80 characters)."""
    text = f"در صف؛ نوبت حدود {local:%H:%M}" + (f" ({local:%m/%d})" if local.date() != today else "")
    text += "، حساب در استراحت است" if paused else f"، فاصلهٔ ضدبن {app['gap_min_minutes']}–{app['gap_max_minutes']} دقیقه"
    return text[:80]


async def _pacing_app(db: AsyncSession, channel: Channel) -> dict[str, Any] | None:
    from app.services.integrations import instagram_app, linkedin_app

    if channel.platform == "instagram" and (channel.config or {}).get("mode") == "private":
        return await instagram_app(db)
    if channel.platform == "linkedin":
        return await linkedin_app(db)
    return None


def _aware(dt: datetime | None) -> datetime | None:
    from datetime import timezone

    return dt.replace(tzinfo=timezone.utc) if dt and not dt.tzinfo else dt


def scheduled_text(local: datetime, today, shifted: bool) -> str:
    text = f"زمان‌بندی‌شده برای {local:%H:%M}" + (f" ({local:%m/%d})" if local.date() != today else "")
    return (text + ("؛ برای فاصلهٔ ضدبن با انتشار دیگر جابه‌جا شد" if shifted else ""))[:80]


async def estimate_turn(db: AsyncSession, channel: Channel, op: Operation, at: datetime | None = None) -> datetime | None:
    """When this new publish runs, set on the operation (run_after and its step text) so the seller sees it at once.

    `at` is the time the seller picked (None = as soon as possible). Accounts that publish on a pace (Instagram login,
    LinkedIn) get the first time at or after it that keeps the anti-ban gap from every other publish waiting on the
    same account (scheduled or not) and stays within the daily cap; so a picked time moves a little when it falls
    too close to another one, and an unscheduled publish can go in between scheduled ones. run_after is a lower
    bound: the worker still checks the pace when it runs. Returns the time, or None for "now"."""
    now = utcnow()
    at = _aware(at)
    if at and at <= now + timedelta(seconds=60):
        at = None
    local_today = (now + timedelta(hours=3, minutes=30)).date()
    app = await _pacing_app(db, channel)
    if not app:
        if at:
            op.run_after = at
            op.current_step = scheduled_text(at + timedelta(hours=3, minutes=30), local_today, False)
        return at
    from app.platforms import instagram_private as igp

    cfg = channel.config or {}
    rows = (await db.execute(select(Operation.run_after).where(
        Operation.type == "publish", Operation.channel_id == channel.id, Operation.status.in_(("queued", "running")),
        Operation.id != op.id))).scalars().all()
    others = sorted(max(_aware(r) or now, now) for r in rows)
    gap = timedelta(minutes=int(app["gap_min_minutes"]))
    limit = int(app["daily_limit"])
    want = at or now
    when = max(want, igp.next_slot(channel, app, now) or now)
    for _ in range(500):
        clash = [o for o in others if abs(o - when) < gap]
        if clash:
            when = max(clash) + gap
            continue
        day = igp._day(when)
        used = sum(1 for o in others if igp._day(o) == day) + (int(cfg.get("count") or 0) if cfg.get("day") == day else 0)
        if used >= limit:  # that (Tehran) day is full: the next day, half an hour after midnight
            nxt = datetime.fromisoformat(day) + timedelta(days=1)
            when = (nxt - timedelta(hours=3, minutes=30)).replace(tzinfo=now.tzinfo) + timedelta(minutes=30)
            continue
        break
    if when <= now + timedelta(seconds=60):
        return None
    local = when + timedelta(hours=3, minutes=30)
    op.run_after = when
    if at:
        op.current_step = scheduled_text(local, local_today, when - at > timedelta(minutes=1))
    else:
        paused = igp._parse(cfg.get("paused_until"))
        op.current_step = turn_text(local, local_today, app, bool(paused and paused > now))
    return when


async def cancel_operation(db: AsyncSession, op: Operation) -> bool:
    """Cancel an operation that has not started. False when the worker already took it."""
    from sqlalchemy import update

    res = await db.execute(update(Operation).where(Operation.id == op.id, Operation.status == "queued")
                           .values(status="cancelled", finished_at=utcnow(), current_step="لغو شد"))
    if not res.rowcount:
        return False
    if op.type == "publish" and op.ref_type == "listing" and op.ref_id:
        listing = await db.get(ChannelListing, op.ref_id)
        if listing and listing.status == "queued":
            listing.status = "published" if listing.remote_id else "not_published"
    await db.flush()
    return True


async def queue_publish(db: AsyncSession, product: Product, channel: Channel, *, trigger: str, actor: str,
                        show_price: bool | None = None, format: str = "post", at: datetime | None = None) -> int | None:
    """Queue one publish. `format` is post | story | reel (story and reel: Instagram only). A post or reel is the
    product's listing on the channel; a story lasts a day, so it keeps no listing and may be posted again."""
    payload: dict[str, Any] = {"show_price": show_price} if show_price is not None else {}
    if format != "post":
        payload["format"] = format
    name = channel.display_name or channel.platform
    if format == "story":
        busy = await db.scalar(select(Operation.id).where(
            Operation.type == "publish", Operation.ref_type == "story", Operation.product_id == product.id,
            Operation.channel_id == channel.id, Operation.status.in_(("queued", "running"))).limit(1))
        if busy:
            return None
        op = await enqueue(db, store_id=product.store_id, type="publish", channel_id=channel.id, product_id=product.id,
                           ref_type="story", trigger=trigger, actor=actor, payload=payload,
                           title_fa=f"استوری «{product.title[:60]}» در {name}")
        await estimate_turn(db, channel, op, at)
        return op.id
    listing = (await db.execute(select(ChannelListing).where(
        ChannelListing.product_id == product.id, ChannelListing.channel_id == channel.id))).scalar_one_or_none()
    if not listing:
        listing = ChannelListing(product_id=product.id, channel_id=channel.id)
        db.add(listing)
        await db.flush()
    if listing.status in ("queued", "publishing"):
        return None  # never run two publishes of the same product × channel at once
    listing.status = "queued"
    op = await enqueue(db, store_id=product.store_id, type="publish", channel_id=channel.id, product_id=product.id,
                       ref_type="listing", ref_id=listing.id, trigger=trigger, actor=actor, payload=payload or None,
                       title_fa=f"انتشار {FORMAT_FA.get(format, '')}«{product.title[:60]}» در {name}")
    await estimate_turn(db, channel, op, at)
    return op.id
