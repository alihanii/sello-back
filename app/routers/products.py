import csv
import io

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import selectinload

from app.deps import DB, CurrentUser, get_store_for
from app.models import Channel, ChannelListing, Operation, Product, ProductImage, Store
from app.schemas import OperationOut, ProductIn, ProductOut, ProductPatch
from app.services.ops import queue_publish
from app.services.media import ALLOWED, save_bytes
from app.services.ops import record_event
from app.services import product_import
from app.services.plans import check_limit, limits_for

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
    await record_event(db, store_id, "product_created")
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
    for k, v in data.items():
        setattr(p, k, v)
    if images is not None:
        p.images.clear()
        await db.flush()
        for i, url in enumerate(images):
            p.images.append(ProductImage(url=url, position=i))
    for l in p.listings:
        if l.status == "published":
            l.status = "out_of_sync"
    await db.commit()
    pid = p.id
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


@router.post("/products/{product_id}/stock", response_model=ProductOut)
async def set_stock(product_id: int, body: StockIn, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    p.stock_qty = body.qty
    await db.commit()
    pid = p.id
    db.expunge_all()
    return await _get(db, user, pid)


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


# ---------- import from Excel / CSV ----------

@router.get("/stores/{store_id}/products/import-template")
async def import_template(store_id: int, user: CurrentUser, db: DB, format: str = "xlsx"):
    await get_store_for(db, user, store_id)
    if format == "csv":
        return Response(product_import.template_csv(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sello-products-template.csv"})
    return Response(product_import.template_xlsx(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=sello-products-template.xlsx"})


async def _skus(db, store_id: int) -> dict[str, int]:
    rows = await db.execute(select(Product.sku, Product.id).where(Product.store_id == store_id, Product.sku != "",
                                                                  Product.status != "archived"))
    return {sku: pid for sku, pid in rows}


@router.post("/stores/{store_id}/products/import/preview")
async def import_preview(store_id: int, user: CurrentUser, db: DB, file: UploadFile = File(...)):
    """Reads the file and checks every row; nothing is saved yet."""
    await get_store_for(db, user, store_id)
    try:
        columns, raw_rows = product_import.read_rows(file.filename or "", await file.read())
    except product_import.ImportError_ as exc:
        raise HTTPException(422, str(exc))
    known = await _skus(db, store_id)
    seen: set[str] = set()
    rows = []
    for raw in raw_rows:
        data, errors = product_import.clean_row(raw)
        sku = data.get("sku", "")
        if sku and sku in seen:
            errors.append("کد کالا در این فایل تکراری است.")
        seen.add(sku) if sku else None
        rows.append({"row": raw["row"], "data": data, "errors": errors,
                     "action": "update" if sku and sku in known else "create"})
    ok = [r for r in rows if not r["errors"]]
    return {"columns": columns, "rows": rows, "total": len(rows), "valid": len(ok),
            "create": sum(r["action"] == "create" for r in ok), "update": sum(r["action"] == "update" for r in ok)}


class ImportItem(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    price: int | None = Field(None, ge=0)
    compare_at_price: int | None = Field(None, ge=0)
    stock_qty: int | None = Field(None, ge=0)
    sku: str = ""
    brand: str | None = None
    category_hint: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    images: list[str] = []


class ImportIn(BaseModel):
    items: list[ImportItem] = Field(max_length=product_import.MAX_ROWS)


async def _fetch_image(client: httpx.AsyncClient, store_id: int, url: str) -> str | None:
    try:
        resp = await client.get(url)
        ctype = resp.headers.get("content-type", "").split(";")[0]
        if resp.status_code != 200 or not ctype.startswith("image/") or len(resp.content) > 8 * 1024 * 1024:
            return None
        return save_bytes(store_id, resp.content, content_type=ctype)
    except httpx.HTTPError:
        return None


@router.post("/stores/{store_id}/products/import")
async def import_products(store_id: int, body: ImportIn, user: CurrentUser, db: DB):
    """Saves the rows checked in the preview: a known SKU updates that product, anything else is a new product."""
    await get_store_for(db, user, store_id)
    known = await _skus(db, store_id)
    new_count = sum(1 for it in body.items if not (it.sku and it.sku in known))
    limits = await limits_for(db, user)
    used = await db.scalar(select(func.count()).select_from(Product).join(Store).where(
        Store.owner_id == user.id, Product.status != "archived"))
    if used + new_count > limits["max_products"]:
        raise HTTPException(402, {"message": f"با این فایل از سقف محصول پلن ({limits['max_products']}) رد می‌شوی؛ "
                                             f"فقط {max(0, limits['max_products'] - used)} محصول جدید دیگر جا دارد.",
                                  "action": {"type": "upgrade", "label": "ارتقای پلن"}})
    created = updated = 0
    image_failed = 0
    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 SelloBot/1.0"}) as client:
        for it in body.items:
            fields = it.model_dump(exclude={"images"}, exclude_none=True)
            pid = known.get(it.sku) if it.sku else None
            if pid:
                p = await db.get(Product, pid)
                for k, v in fields.items():
                    setattr(p, k, v)
                for l in (await db.execute(select(ChannelListing).where(ChannelListing.product_id == pid))).scalars():
                    if l.status == "published":
                        l.status = "out_of_sync"
                updated += 1
            else:
                p = Product(store_id=store_id, source="manual", created_by="import", status="ready", **fields)
                db.add(p)
                await db.flush()
                if it.sku:
                    known[it.sku] = p.id
                created += 1
            if it.images:
                start = await db.scalar(select(func.count()).select_from(ProductImage).where(ProductImage.product_id == p.id))
                for i, url in enumerate(it.images[:5]):
                    local = await _fetch_image(client, store_id, url)
                    if local:
                        db.add(ProductImage(product_id=p.id, url=local, position=start + i, origin_url=url))
                    else:
                        image_failed += 1
    if created:
        await record_event(db, store_id, "product_created", created)
    await db.commit()
    return {"created": created, "updated": updated, "image_failed": image_failed}
