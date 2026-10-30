import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from app.core.db import SessionLocal
from app.core.security import decode_token
from app.deps import DB, CurrentUser, get_store_for
from app.models import ChannelListing, CrawlItem, Operation, Store, User
from app.schemas import OperationDetail, OperationOut

router = APIRouter(tags=["operations"])


@router.get("/stores/{store_id}/operations")
async def list_operations(store_id: int, user: CurrentUser, db: DB, type: str = "", status: str = "",
                          channel_id: int | None = None, product_id: int | None = None, page: int = 1, size: int = 30):
    await get_store_for(db, user, store_id)
    q = select(Operation).where(Operation.store_id == store_id)
    if type:
        q = q.where(Operation.type.in_(type.split(",")))
    if status:
        q = q.where(Operation.status.in_(status.split(",")))
    if channel_id:
        q = q.where(Operation.channel_id == channel_id)
    if product_id:
        q = q.where(Operation.product_id == product_id)
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    rows = (await db.execute(q.order_by(Operation.id.desc()).offset((page - 1) * size).limit(size))).scalars().all()
    return {"items": [OperationOut.model_validate(r) for r in rows], "total": total}


@router.get("/stores/{store_id}/operations/summary")
async def summary(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    running = await db.scalar(select(func.count()).select_from(Operation).where(
        Operation.store_id == store_id, Operation.status.in_(["queued", "running"])))
    unseen_failed = await db.scalar(select(func.count()).select_from(Operation).where(
        Operation.store_id == store_id, Operation.status.in_(["failed", "partial"]), ~Operation.seen))
    return {"running": running, "unseen_failed": unseen_failed}


@router.post("/stores/{store_id}/operations/mark-seen")
async def mark_seen(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    await db.execute(update(Operation).where(Operation.store_id == store_id, ~Operation.seen,
                                             Operation.status.in_(["failed", "partial", "succeeded"])).values(seen=True))
    await db.commit()
    return {"ok": True}


@router.get("/operations/{op_id}", response_model=OperationDetail)
async def get_operation(op_id: int, user: CurrentUser, db: DB):
    op = (await db.execute(select(Operation).where(Operation.id == op_id)
                           .options(selectinload(Operation.steps)))).scalar_one_or_none()
    if not op:
        raise HTTPException(404, "عملیات پیدا نشد.")
    await get_store_for(db, user, op.store_id)
    return op


@router.get("/operations/{op_id}/crawl-items")
async def crawl_items(op_id: int, user: CurrentUser, db: DB):
    op = await db.get(Operation, op_id)
    if not op:
        raise HTTPException(404, "عملیات پیدا نشد.")
    await get_store_for(db, user, op.store_id)
    rows = (await db.execute(select(CrawlItem).where(CrawlItem.operation_id == op_id).order_by(CrawlItem.id))).scalars()
    return [{"id": r.id, "external_id": r.external_id, "raw": r.raw, "extracted": r.extracted, "product_id": r.product_id,
             "result": r.result, "error_fa": r.error_fa} for r in rows]


@router.post("/operations/{op_id}/retry", response_model=OperationOut, status_code=202)
async def retry(op_id: int, user: CurrentUser, db: DB):
    op = await db.get(Operation, op_id)
    if not op:
        raise HTTPException(404, "عملیات پیدا نشد.")
    await get_store_for(db, user, op.store_id)
    if op.status in ("queued", "running"):
        raise HTTPException(409, "این عملیات هنوز در حال اجراست.")
    if op.type == "connect":
        raise HTTPException(400, "برای اتصال دوباره از صفحهٔ کانال‌ها اقدام کنید.")
    if op.type == "publish" and op.ref_id:
        listing = await db.get(ChannelListing, op.ref_id)
        if listing.status in ("queued", "publishing"):
            raise HTTPException(409, "انتشار این محصول در این کانال در حال اجراست.")
        listing.status = "queued"
    new = Operation(store_id=op.store_id, type=op.type, title_fa=op.title_fa, channel_id=op.channel_id,
                    product_id=op.product_id, ref_type=op.ref_type, ref_id=op.ref_id, payload=op.payload,
                    trigger="retry", actor="user", attempt=op.attempt + 1, status="queued")
    db.add(new)
    await db.commit()
    return new


@router.get("/stores/{store_id}/operations/stream")
async def stream(store_id: int, request: Request, token: str):
    """SSE: pushes operations that changed since the last tick. EventSource can't send headers,
    so the JWT comes as a query param."""
    try:
        uid = int(decode_token(token)["sub"])
    except Exception:
        raise HTTPException(401, "نشست نامعتبر.")
    async with SessionLocal() as db:
        user = await db.get(User, uid)
        store = await db.get(Store, store_id)
        if not user or not store or (store.owner_id != uid and user.role != "admin"):
            raise HTTPException(404, "فروشگاه پیدا نشد.")

    async def events():
        last = None
        while not await request.is_disconnected():
            async with SessionLocal() as db:
                q = select(Operation).where(Operation.store_id == store_id)
                q = q.where(Operation.updated_at > last) if last else q.order_by(Operation.id.desc()).limit(1)
                rows = (await db.execute(q.order_by(Operation.updated_at))).scalars().all()
            for r in rows:
                last = max(last, r.updated_at) if last else r.updated_at
                yield f"data: {json.dumps(OperationOut.model_validate(r).model_dump(mode='json'), ensure_ascii=False)}\n\n"
            if last is None:
                from app.core.db import utcnow
                last = utcnow()
            yield ": ping\n\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
