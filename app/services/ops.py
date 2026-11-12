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


async def _pacing_app(db: AsyncSession, channel: Channel) -> dict[str, Any] | None:
    from app.services.integrations import instagram_app, linkedin_app

    if channel.platform == "instagram" and (channel.config or {}).get("mode") == "private":
        return await instagram_app(db)
    if channel.platform == "linkedin":
        return await linkedin_app(db)
    return None


async def estimate_turn(db: AsyncSession, channel: Channel, op: Operation) -> datetime | None:
    """For accounts that publish on a pace (Instagram login, LinkedIn): when this new publish will roughly run,
    behind the ones already waiting on the same account. Sets the operation's run_after (a lower bound; the worker
    still checks the pace when it runs) and its step text, so the seller sees the turn right away. None = now."""
    app = await _pacing_app(db, channel)
    if not app:
        return None
    from app.platforms import instagram_private as igp

    now = utcnow()
    cfg = channel.config or {}
    ahead = int(await db.scalar(select(func.count()).select_from(Operation).where(
        Operation.type == "publish", Operation.channel_id == channel.id, Operation.status.in_(("queued", "running")),
        Operation.id != op.id)) or 0)
    when = igp.next_slot(channel, app, now) or now
    count = int(cfg.get("count") or 0) if cfg.get("day") == igp._day(when) else 0
    gap = timedelta(minutes=int(app["gap_min_minutes"]))
    for _ in range(ahead):
        count += 1
        when = when + gap
        if count >= int(app["daily_limit"]):  # the cap is reached: the next one goes the next (Tehran) day
            day = datetime.fromisoformat(igp._day(when)) + timedelta(days=1)
            when = max(when, (day - timedelta(hours=3, minutes=30)).replace(tzinfo=now.tzinfo) + timedelta(minutes=30))
            count = 0
    if when <= now + timedelta(seconds=60):
        return None
    local, today = when + timedelta(hours=3, minutes=30), (now + timedelta(hours=3, minutes=30)).date()
    op.run_after = when
    reason = "حساب موقتاً استراحت می‌کند" if igp._parse(cfg.get("paused_until")) and igp._parse(cfg.get("paused_until")) > now \
        else f"برای جلوگیری از بن، بین انتشارهای هر حساب {app['gap_min_minutes']} تا {app['gap_max_minutes']} دقیقه فاصله است"
    op.current_step = (f"در صف؛ نوبت انتشار حدود {local:%H:%M}" + (f" ({local:%m/%d})" if local.date() != today else "")
                       + f"، چون {reason}")
    return when


async def queue_publish(db: AsyncSession, product: Product, channel: Channel, *, trigger: str, actor: str,
                        show_price: bool | None = None, format: str = "post") -> int | None:
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
        await estimate_turn(db, channel, op)
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
    await estimate_turn(db, channel, op)
    return op.id
