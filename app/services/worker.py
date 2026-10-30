"""In-process worker. Claims queued operations with a conditional UPDATE so several replicas
never run the same one (pattern from platform-wrappers' messenger_queue), plus a scheduler
for crawl sources and scheduled automations."""

import asyncio
import logging
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.db import SessionLocal, utcnow
from app.models import Channel, CrawlSource, Operation
from app.services import automation
from app.services.jobs import HANDLERS
from app.services.ops import OpContext, enqueue, fail_operation

log = logging.getLogger("sello.worker")
SCHEDULE_DELTA = {"hourly": timedelta(hours=1), "daily": timedelta(days=1), "weekly": timedelta(weeks=1)}


async def _claim() -> int | None:
    async with SessionLocal() as db:
        candidate = await db.scalar(select(Operation.id).where(Operation.status == "queued").order_by(Operation.id).limit(1))
        if candidate is None:
            return None
        res = await db.execute(update(Operation).where(Operation.id == candidate, Operation.status == "queued")
                               .values(status="running", started_at=utcnow()))
        await db.commit()
        return candidate if res.rowcount == 1 else None


async def run_operation(op_id: int) -> None:
    async with SessionLocal() as db:
        op = (await db.execute(select(Operation).where(Operation.id == op_id)
                               .options(selectinload(Operation.steps)))).scalar_one()
        ctx = OpContext(db, op)
        handler = HANDLERS.get(op.type)
        try:
            if not handler:
                raise RuntimeError(f"no handler for {op.type}")
            await handler(ctx)
            if op.status == "running":
                op.status = "succeeded"
            op.progress, op.finished_at, op.current_step = 100, utcnow(), ""
            await db.commit()
        except Exception as exc:  # noqa: BLE001 - every failure is reported on the operation
            if not hasattr(exc, "message_fa"):
                log.exception("operation %s crashed", op_id)
            await db.rollback()
            op = await db.get(Operation, op_id)
            fail_operation(op, exc)
            await _sync_failure(db, op)
            await db.commit()


async def _sync_failure(db, op: Operation) -> None:
    """The handler's own writes were rolled back; re-apply the visible failure state."""
    from app.models import ChannelListing
    from app.services.notify import notify
    from app.services.ops import record_event

    if op.type == "publish" and op.ref_id:
        listing = await db.get(ChannelListing, op.ref_id)
        if listing:
            listing.status, listing.last_error_fa = "failed", op.error_message_fa
        await record_event(db, op.store_id, "publish_failed", operation_id=op.id, channel_id=op.channel_id)
        await notify(db, op.store_id, "publish_failed", "انتشار ناموفق: " + (op.title_fa or ""), op.error_message_fa or "",
                     link=f"/app/s/{op.store_id}/activity/{op.id}")
    elif op.type == "ai_post":
        await notify(db, op.store_id, "publish_failed", "ساخت پست با AI ناموفق بود", op.error_message_fa or "",
                     link=f"/app/s/{op.store_id}/activity/{op.id}")
    elif op.type == "crawl":
        await record_event(db, op.store_id, "crawl_failed", operation_id=op.id, channel_id=op.channel_id)
    elif op.type in ("connect", "health_check") and op.channel_id:
        ch = await db.get(Channel, op.channel_id)
        if ch:
            ch.status = "expired" if op.error_code in ("expired", "invalid_token", "bot_not_admin") else "error"
            ch.last_error = op.error_message_fa
            await notify(db, op.store_id, "channel_error", f"مشکل در کانال «{ch.display_name or ch.platform}»",
                         op.error_message_fa or "", link=f"/app/s/{op.store_id}/channels")
    op.payload = {} if op.type == "connect" else op.payload


async def _scheduler_tick() -> None:
    async with SessionLocal() as db:
        now = utcnow()
        sources = (await db.execute(select(CrawlSource).where(CrawlSource.is_active, CrawlSource.schedule != "manual"))).scalars()
        for src in sources:
            due = src.next_run_at is None or src.next_run_at.replace(tzinfo=now.tzinfo) <= now
            if not due:
                continue
            await enqueue(db, store_id=src.store_id, type="crawl", ref_type="crawl_source", ref_id=src.id,
                          channel_id=src.channel_id, trigger="schedule", actor="ai", title_fa=f"کرال زمان‌بندی‌شدهٔ «{src.name}»")
            src.next_run_at = now + SCHEDULE_DELTA[src.schedule]
        await automation.run_scheduled(db)
        await automation.run_ai_content(db)
        await db.commit()


async def worker_loop(stop: asyncio.Event) -> None:
    sem = asyncio.Semaphore(settings.worker_concurrency)
    tasks: set[asyncio.Task] = set()
    last_tick = 0.0
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        try:
            if loop.time() - last_tick > 60:
                last_tick = loop.time()
                await _scheduler_tick()
            while sem._value > 0:  # noqa: SLF001
                op_id = await _claim()
                if op_id is None:
                    break
                await sem.acquire()
                t = asyncio.create_task(run_operation(op_id))
                t.add_done_callback(lambda _t: (sem.release(), tasks.discard(_t)))
                tasks.add(t)
        except Exception:  # noqa: BLE001
            log.exception("worker loop error")
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
        except asyncio.TimeoutError:
            pass
