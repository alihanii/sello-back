"""Status center plumbing: every action is an Operation with ordered steps. Handlers run
inside an OpContext; `ctx.step()` times the step and stores its (redacted) request/response."""

import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import utcnow
from app.core.security import redact
from app.models import ActivityEvent, Operation, OperationStep
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

    async def progress(self, pct: int, label_fa: str | None = None) -> None:
        self.op.progress = max(0, min(100, pct))
        if label_fa:
            self.op.current_step = label_fa
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
