import csv
import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import selectinload

from app.deps import DB, CurrentUser, get_store_for
from app.models import Channel, ChannelListing, InventoryMovement, Operation, Product, ProductImage
from app.schemas import OperationOut, ProductIn, ProductOut, ProductPatch
from app.services import automation
from app.services.automation import queue_publish
from app.services.media import ALLOWED, save_bytes
from app.services.ai import route_problem
from app.services.ops import enqueue, record_event
from app.services.plans import check_limit

router = APIRouter(tags=["products"])


async def _get(db, user, product_id: int) -> Product:
    p = (await db.execute(select(Product).where(Product.id == product_id)
                          .options(selectinload(Product.images), selectinload(Product.listings)))).scalar_one_or_none()
    if not p:
        raise HTTPException(404, "محصول پیدا نشد.")
    await get_store_for(db, user, p.store_id)
    return p


@router.get("/stores/{store_id}/products")
async def list_products(
    store_id: int, user: CurrentUser, db: DB,
    q: str = "", source: str = "", status: str = "", stock: str = "", tag: str = "",
    published_in: int | None = None, not_published_in: int | None = None,
    page: int = 1, size: int = Query(25, le=200), sort: str = "-created_at",
):
    await get_store_for(db, user, store_id)
    stmt = select(Product).where(Product.store_id == store_id)
    stmt = stmt.where(Product.status == status) if status else stmt.where(Product.status != "archived")
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Product.title.ilike(like), Product.sku.ilike(like), Product.description.ilike(like)))
    if source:
        stmt = stmt.where(Product.source == source)
    if stock == "out":
        stmt = stmt.where(Product.stock_qty <= 0)
    elif stock == "low":
        stmt = stmt.where(Product.stock_qty > 0, Product.stock_qty <= 3)
    if tag:
        stmt = stmt.where(cast(Product.tags, String).like(f'%"{tag}"%'))
    pub = select(ChannelListing.product_id).where(ChannelListing.status == "published")
    if published_in:
        stmt = stmt.where(Product.id.in_(pub.where(ChannelListing.channel_id == published_in)))
    if not_published_in:
        stmt = stmt.where(Product.id.not_in(pub.where(ChannelListing.channel_id == not_published_in)))
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    col = {"created_at": Product.created_at, "price": Product.price, "stock": Product.stock_qty,
           "title": Product.title}.get(sort.lstrip("-"), Product.created_at)
    stmt = stmt.order_by(col.desc() if sort.startswith("-") else col.asc(), Product.id.desc())
    stmt = stmt.options(selectinload(Product.images), selectinload(Product.listings)).offset((page - 1) * size).limit(size)
    items = (await db.execute(stmt)).scalars().all()
    return {"items": [ProductOut.model_validate(p) for p in items], "total": total, "page": page, "size": size}


@router.post("/stores/{store_id}/products", response_model=ProductOut)
async def create_product(store_id: int, body: ProductIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    await check_limit(db, user, "products")
    data = body.model_dump(exclude={"images"})
    p = Product(store_id=store_id, source="manual", created_by="user", **data)
    db.add(p)
    await db.flush()
    for i, url in enumerate(body.images):
        db.add(ProductImage(product_id=p.id, url=url, position=i))
    if p.stock_qty:
        db.add(InventoryMovement(product_id=p.id, delta=p.stock_qty, qty_after=p.stock_qty, reason="manual", note="موجودی اولیه"))
    await record_event(db, store_id, "product_created")
    await db.commit()
    await automation.fire(db, store_id, "product_created", p.id)
    await db.commit()
    return await _get(db, user, p.id)


@router.get("/products/{product_id}", response_model=ProductOut)
async def get_product(product_id: int, user: CurrentUser, db: DB):
    return await _get(db, user, product_id)


@router.patch("/products/{product_id}", response_model=ProductOut)
async def update_product(product_id: int, body: ProductPatch, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    data = body.model_dump(exclude_unset=True)
    images = data.pop("images", None)
    old_stock = p.stock_qty
    for k, v in data.items():
        setattr(p, k, v)
    if images is not None:
        p.images.clear()
        await db.flush()
        for i, url in enumerate(images):
            p.images.append(ProductImage(url=url, position=i))
    if p.stock_qty != old_stock:
        db.add(InventoryMovement(product_id=p.id, delta=p.stock_qty - old_stock, qty_after=p.stock_qty, reason="manual"))
    for l in p.listings:
        if l.status == "published":
            l.status = "out_of_sync"
    await db.commit()
    trigger = "stock_changed" if set(data) == {"stock_qty"} else "product_updated"
    pid, store_id = p.id, p.store_id  # read before expire_all: lazy loads aren't allowed in async sessions
    await automation.fire(db, store_id, trigger, pid)
    await db.commit()
    db.expunge_all()  # reload relationships fresh; expire_all would force lazy loads on `user`
    return await _get(db, user, pid)


@router.delete("/products/{product_id}")
async def archive_product(product_id: int, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    p.status = "archived"
    await db.commit()
    return {"ok": True}


class StockIn(BaseModel):
    qty: int
    note: str = ""


@router.post("/products/{product_id}/stock", response_model=ProductOut)
async def set_stock(product_id: int, body: StockIn, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    db.add(InventoryMovement(product_id=p.id, delta=body.qty - p.stock_qty, qty_after=body.qty, reason="manual", note=body.note))
    p.stock_qty = body.qty
    await db.commit()
    pid = p.id
    await automation.fire(db, p.store_id, "stock_changed", pid)
    await db.commit()
    db.expunge_all()
    return await _get(db, user, pid)


@router.get("/products/{product_id}/inventory")
async def inventory_history(product_id: int, user: CurrentUser, db: DB):
    await _get(db, user, product_id)
    rows = (await db.execute(select(InventoryMovement).where(InventoryMovement.product_id == product_id)
                             .order_by(InventoryMovement.id.desc()).limit(100))).scalars()
    return [{"id": r.id, "delta": r.delta, "qty_after": r.qty_after, "reason": r.reason, "note": r.note,
             "created_at": r.created_at} for r in rows]


@router.post("/stores/{store_id}/uploads")
async def upload_images(store_id: int, user: CurrentUser, db: DB, files: list[UploadFile] = File(...)):
    await get_store_for(db, user, store_id)
    urls = []
    for f in files[:10]:
        if f.content_type not in ALLOWED:
            raise HTTPException(422, "فقط عکس JPG، PNG، WEBP یا GIF مجاز است.")
        data = await f.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(422, "حجم هر عکس حداکثر ۱۰ مگابایت است.")
        urls.append(save_bytes(store_id, data, f.filename or "", f.content_type or ""))
    return {"urls": urls}


class PublishIn(BaseModel):
    channel_ids: list[int]


@router.post("/products/{product_id}/publish", response_model=list[OperationOut], status_code=202)
async def publish(product_id: int, body: PublishIn, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    op_ids = []
    for cid in body.channel_ids:
        ch = await db.get(Channel, cid)
        if not ch or ch.store_id != p.store_id:
            raise HTTPException(404, "کانال پیدا نشد.")
        op_id = await queue_publish(db, p, ch, trigger="user", actor="user")
        if op_id:
            op_ids.append(op_id)
    await db.commit()
    return (await db.execute(select(Operation).where(Operation.id.in_(op_ids)))).scalars().all()


class BulkIn(BaseModel):
    product_ids: list[int]
    action: str  # publish | tag | untag | archive | status
    channel_ids: list[int] = []
    tag: str = ""
    status: str = ""


@router.post("/stores/{store_id}/products/bulk")
async def bulk(store_id: int, body: BulkIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    products = (await db.execute(select(Product).where(Product.store_id == store_id, Product.id.in_(body.product_ids))
                                 .options(selectinload(Product.images)))).scalars().all()
    queued = 0
    for p in products:
        if body.action == "publish":
            for cid in body.channel_ids:
                ch = await db.get(Channel, cid)
                if ch and ch.store_id == store_id and await queue_publish(db, p, ch, trigger="user", actor="user"):
                    queued += 1
        elif body.action == "tag" and body.tag:
            p.tags = sorted(set(p.tags or []) | {body.tag})
        elif body.action == "untag":
            p.tags = [t for t in p.tags or [] if t != body.tag]
        elif body.action == "archive":
            p.status = "archived"
        elif body.action == "status" and body.status in ("draft", "ready"):
            p.status = body.status
    await db.commit()
    return {"ok": True, "affected": len(products), "queued": queued}


@router.get("/stores/{store_id}/products.csv")
async def export_csv(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(Product).where(Product.store_id == store_id, Product.status != "archived"))).scalars()
    buf = io.StringIO()
    buf.write("﻿")
    w = csv.writer(buf)
    w.writerow(["id", "sku", "title", "price", "stock", "source", "status", "tags"])
    for p in rows:
        w.writerow([p.id, p.sku, p.title, p.price, p.stock_qty, p.source, p.status, " ".join(p.tags or [])])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=products-{store_id}.csv"})


class CleanImageIn(BaseModel):
    image_id: int | None = None  # empty: generate a photo from the product text
    category: str = "general"  # flat_clothing | modeled_clothing | footwear | general
    style_images: list[str] = []  # style samples; empty = the shop's earlier products
    style_product_ids: list[int] = []


@router.post("/products/{product_id}/clean-image", response_model=OperationOut, status_code=202)
async def clean_image(product_id: int, body: CleanImageIn, user: CurrentUser, db: DB):
    """Redraws a product photo as a clean white-background main image with AI."""
    p = await _get(db, user, product_id)
    if body.image_id and not any(i.id == body.image_id for i in p.images):
        raise HTTPException(404, "عکس پیدا نشد.")
    if problem := await route_problem("image_edit"):
        raise HTTPException(422, problem)
    op = await enqueue(db, store_id=p.store_id, type="image_clean", product_id=p.id, actor="ai",
                       title_fa=f"{'تمیز کردن' if body.image_id else 'ساخت'} عکس «{p.title[:60]}»",
                       payload={"image_id": body.image_id, "category": body.category,
                                "style_images": body.style_images[:3], "style_product_ids": body.style_product_ids[:3]})
    await db.commit()
    return op
