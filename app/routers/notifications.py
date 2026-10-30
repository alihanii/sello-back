from fastapi import APIRouter, Query
from sqlalchemy import func, select, update

from app.deps import DB, CurrentUser
from app.models import Notification

router = APIRouter(tags=["notifications"])


@router.get("/notifications")
async def list_notifications(user: CurrentUser, db: DB, store_id: int | None = None, limit: int = 30):
    q = select(Notification).where(Notification.user_id == user.id)
    if store_id:
        q = q.where(Notification.store_id == store_id)
    rows = (await db.execute(q.order_by(Notification.id.desc()).limit(min(limit, 100)))).scalars().all()
    unread = await db.scalar(select(func.count()).select_from(Notification).where(
        Notification.user_id == user.id, Notification.is_read.is_(False),
        *([Notification.store_id == store_id] if store_id else [])))
    return {"unread": unread or 0, "items": [
        {c: getattr(n, c) for c in ("id", "store_id", "kind", "title_fa", "body_fa", "link", "image_url", "is_read", "created_at")}
        for n in rows]}


@router.post("/notifications/read")
async def mark_read(user: CurrentUser, db: DB, ids: list[int] | None = Query(None)):
    q = update(Notification).where(Notification.user_id == user.id, Notification.is_read.is_(False))
    if ids:
        q = q.where(Notification.id.in_(ids))
    await db.execute(q.values(is_read=True))
    await db.commit()
    return {"ok": True}
