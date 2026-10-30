from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import settings
from app.deps import DB, CurrentUser, get_store_for
from app.models import Channel, Operation
from app.platforms.base import PlatformError
from app.platforms.digikala import digikala
from app.platforms.registry import PLATFORM_INFO
from app.schemas import ChannelOut, OperationOut
from app.services.ops import OpContext, enqueue, fail_operation
from app.services.plans import check_limit

router = APIRouter(tags=["channels"])


@router.get("/platforms")
async def platforms():
    return [{"platform": k, **v, "enabled": not v.get("flag") or settings.instagram_enabled} for k, v in PLATFORM_INFO.items()]


@router.get("/stores/{store_id}/channels", response_model=list[ChannelOut])
async def list_channels(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    q = select(Channel).where(Channel.store_id == store_id, Channel.status != "disconnected").order_by(Channel.id)
    return (await db.execute(q)).scalars().all()


class ConnectIn(BaseModel):
    platform: str
    channel_id: int | None = None  # reconnect an existing channel
    data: dict[str, Any] = {}


async def _channel(db, user, store_id: int, body: ConnectIn) -> Channel:
    if body.platform not in PLATFORM_INFO:
        raise HTTPException(422, "پلتفرم پشتیبانی نمی‌شود.")
    if body.channel_id:
        ch = await db.get(Channel, body.channel_id)
        if not ch or ch.store_id != store_id:
            raise HTTPException(404, "کانال پیدا نشد.")
        return ch
    await check_limit(db, user, "channels")
    ch = Channel(store_id=store_id, platform=body.platform, status="disconnected", config={})
    db.add(ch)
    await db.flush()
    return ch


@router.post("/stores/{store_id}/channels/digikala/send-code")
async def digikala_send_code(store_id: int, body: ConnectIn, user: CurrentUser, db: DB):
    """Step 1 of Digikala OTP; runs inline because the seller is waiting for the SMS."""
    await get_store_for(db, user, store_id)
    body.platform = "digikala"
    ch = await _channel(db, user, store_id, body)
    op = await enqueue(db, store_id=store_id, type="connect", channel_id=ch.id, title_fa="ارسال کد تأیید دیجی‌کالا")
    op.status = "running"
    await db.commit()
    ctx = OpContext(db, op)
    try:
        pending = await digikala.send_code(ctx, body.data.get("phone", ""))
    except PlatformError as exc:
        fail_operation(op, exc)
        await db.commit()
        raise HTTPException(400, {"message": exc.message_fa, "operation_id": op.id})
    ch.config = {**ch.config, "pending": pending}
    op.status = "succeeded"
    await db.commit()
    return {"channel_id": ch.id, "operation_id": op.id}


@router.post("/stores/{store_id}/channels/connect", response_model=OperationOut, status_code=202)
async def connect(store_id: int, body: ConnectIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    ch = await _channel(db, user, store_id, body)
    label = PLATFORM_INFO[body.platform]["label"]
    op = await enqueue(db, store_id=store_id, type="connect", channel_id=ch.id, payload={"data": body.data},
                       title_fa=f"اتصال {label}")
    await db.commit()
    return op


@router.post("/channels/{channel_id}/health", response_model=OperationOut, status_code=202)
async def health(channel_id: int, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    op = await enqueue(db, store_id=ch.store_id, type="health_check", channel_id=ch.id,
                       title_fa=f"بررسی سلامت {ch.display_name or ch.platform}")
    await db.commit()
    return op


class ChannelPatch(BaseModel):
    display_name: str | None = None


@router.patch("/channels/{channel_id}", response_model=ChannelOut)
async def update_channel(channel_id: int, body: ChannelPatch, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    if body.display_name is not None:
        ch.display_name = body.display_name
    await db.commit()
    await db.refresh(ch)
    return ch


@router.delete("/channels/{channel_id}")
async def disconnect(channel_id: int, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    ch.status, ch.credentials_encrypted = "disconnected", None
    await db.commit()
    return {"ok": True}
