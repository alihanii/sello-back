from datetime import timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.models import Channel, CrawlSource, Product
from app.schemas import OperationOut, ProductOut
from app.services import automation
from app.services.ops import enqueue

router = APIRouter(tags=["crawl"])
SCHEDULES = {"manual", "hourly", "daily", "weekly"}


class SourceIn(BaseModel):
    name: str = ""
    type: str  # channel_posts | marketplace_listing | url
    channel_id: int | None = None
    url: str = ""
    schedule: str = "manual"
    extraction_mode: str = "auto"
    is_active: bool = True


def _out(s: CrawlSource) -> dict:
    return {c: getattr(s, c) for c in ("id", "store_id", "name", "type", "channel_id", "url", "schedule",
                                         "extraction_mode", "is_active", "last_run_at", "next_run_at", "created_at")}


@router.get("/stores/{store_id}/crawl-sources")
async def list_sources(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(CrawlSource).where(CrawlSource.store_id == store_id).order_by(CrawlSource.id))).scalars()
    return [_out(s) for s in rows]


@router.post("/stores/{store_id}/crawl-sources")
async def create_source(store_id: int, body: SourceIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    if body.schedule not in SCHEDULES:
        raise HTTPException(422, "زمان‌بندی نامعتبر است.")
    if body.type in ("channel_posts", "marketplace_listing"):
        ch = await db.get(Channel, body.channel_id or 0)
        if not ch or ch.store_id != store_id:
            raise HTTPException(422, "کانال منبع را انتخاب کنید.")
        name = body.name or f"{ch.display_name or ch.platform}"
    elif body.type == "url":
        if not body.url.startswith(("http://", "https://")):
            raise HTTPException(422, "لینک معتبر وارد کنید.")
        name = body.name or body.url[:80]
    else:
        raise HTTPException(422, "نوع منبع نامعتبر است.")
    s = CrawlSource(store_id=store_id, **{**body.model_dump(), "name": name},
                    next_run_at=utcnow() + timedelta(minutes=1) if body.schedule != "manual" else None)
    db.add(s)
    await db.commit()
    return _out(s)


@router.patch("/crawl-sources/{source_id}")
async def update_source(source_id: int, body: SourceIn, user: CurrentUser, db: DB):
    s = await db.get(CrawlSource, source_id)
    if not s:
        raise HTTPException(404, "منبع پیدا نشد.")
    await get_store_for(db, user, s.store_id)
    s.name, s.schedule, s.extraction_mode, s.is_active = body.name or s.name, body.schedule, body.extraction_mode, body.is_active
    await db.commit()
    return _out(s)


@router.delete("/crawl-sources/{source_id}")
async def delete_source(source_id: int, user: CurrentUser, db: DB):
    s = await db.get(CrawlSource, source_id)
    if not s:
        raise HTTPException(404, "منبع پیدا نشد.")
    await get_store_for(db, user, s.store_id)
    await db.delete(s)
    await db.commit()
    return {"ok": True}


@router.post("/crawl-sources/{source_id}/run", response_model=OperationOut, status_code=202)
async def run_now(source_id: int, user: CurrentUser, db: DB):
    s = await db.get(CrawlSource, source_id)
    if not s:
        raise HTTPException(404, "منبع پیدا نشد.")
    await get_store_for(db, user, s.store_id)
    op = await enqueue(db, store_id=s.store_id, type="crawl", ref_type="crawl_source", ref_id=s.id,
                       channel_id=s.channel_id, title_fa=f"کرال «{s.name}»")
    await db.commit()
    return op


@router.get("/stores/{store_id}/crawl-inbox", response_model=list[ProductOut])
async def inbox(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    q = (select(Product).where(Product.store_id == store_id, Product.status == "draft")
         .options(selectinload(Product.images), selectinload(Product.listings)).order_by(Product.id.desc()))
    return (await db.execute(q)).scalars().all()


class InboxAction(BaseModel):
    product_ids: list[int]
    approve: bool


@router.post("/stores/{store_id}/crawl-inbox")
async def inbox_action(store_id: int, body: InboxAction, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(Product).where(Product.store_id == store_id, Product.id.in_(body.product_ids)))).scalars().all()
    for p in rows:
        p.status = "ready" if body.approve else "archived"
    await db.commit()
    if body.approve:
        for p in rows:
            await automation.fire(db, store_id, "product_created", p.id)
        await db.commit()
    return {"ok": True, "count": len(rows)}
