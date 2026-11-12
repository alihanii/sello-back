import csv
import io
import uuid

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import selectinload

from app.deps import DB, CurrentUser, get_store_for
from app.platforms.base import PlatformError
from app.models import Channel, ChannelListing, Operation, Product, ProductImage, Store
from app.schemas import OperationOut, ProductIn, ProductOut, ProductPatch
from app.services.ops import queue_publish
from app.services.media import ALLOWED, VIDEO_TYPES, put, save_bytes
from app.services.ops import record_event
from app.services.text import normalize_fa
from app.services import categories, product_code, product_import, video
from app.services.plans import check_limit, limits_for

router = APIRouter(tags=["products"])


async def _get(db, user, product_id: int) -> Product:
    p = (await db.execute(select(Product).where(Product.id == product_id)
                          .options(selectinload(Product.images), selectinload(Product.listings)))).scalar_one_or_none()
    if not p:
        raise HTTPException(404, "محصول پیدا نشد.")
    await get_store_for(db, user, p.store_id)
    return p


class ProductFilters(BaseModel):
    """The product list's filters; bulk actions reuse them for "every product matching this filter"."""
    q: str = ""
    source: str = ""
    status: str = ""
    stock: str = ""
    tag: str = ""
    category: str | None = None  # "" (sent as "-" in the query string) = products without a category
    published_in: int | None = None
    not_published_in: int | None = None


def _filtered(store_id: int, f: ProductFilters):
    stmt = select(Product).where(Product.store_id == store_id)
    stmt = stmt.where(Product.status == f.status) if f.status else stmt.where(Product.status != "archived")
    if f.q:
        like = f"%{f.q}%"
        stmt = stmt.where(or_(Product.title.ilike(like), Product.sku.ilike(like), Product.description.ilike(like)))
    if f.source:
        stmt = stmt.where(Product.source == f.source)
    if f.stock == "out":
        stmt = stmt.where(Product.stock_qty <= 0)
    elif f.stock == "low":
        stmt = stmt.where(Product.stock_qty > 0, Product.stock_qty <= 3)
    if f.tag:
        stmt = stmt.where(cast(Product.tags, String).like(f'%"{f.tag}"%'))
    if f.category is not None:
        stmt = (stmt.where(or_(Product.category.is_(None), Product.category == "")) if f.category == ""
                else stmt.where(Product.category == f.category))
    pub = select(ChannelListing.product_id).where(ChannelListing.status == "published")
    if f.published_in:
        stmt = stmt.where(Product.id.in_(pub.where(ChannelListing.channel_id == f.published_in)))
    if f.not_published_in:
        stmt = stmt.where(Product.id.not_in(pub.where(ChannelListing.channel_id == f.not_published_in)))
    return stmt


@router.get("/stores/{store_id}/products")
async def list_products(
    store_id: int, user: CurrentUser, db: DB,
    q: str = "", source: str = "", status: str = "", stock: str = "", tag: str = "", category: str | None = None,
    published_in: int | None = None, not_published_in: int | None = None,
    page: int = 1, size: int = Query(25, le=200), sort: str = "-created_at",
):
    await get_store_for(db, user, store_id)
    stmt = _filtered(store_id, ProductFilters(
        q=q, source=source, status=status, stock=stock, tag=tag, category="" if category == "-" else (category or None),
        published_in=published_in, not_published_in=not_published_in))
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    col = {"created_at": Product.created_at, "price": Product.price, "stock": Product.stock_qty,
           "title": Product.title}.get(sort.lstrip("-"), Product.created_at)
    stmt = stmt.order_by(col.desc() if sort.startswith("-") else col.asc(), Product.id.desc())
    stmt = stmt.options(selectinload(Product.images), selectinload(Product.listings)).offset((page - 1) * size).limit(size)
    items = (await db.execute(stmt)).scalars().all()
    return {"items": [ProductOut.model_validate(p) for p in items], "total": total, "page": page, "size": size}


@router.get("/stores/{store_id}/product-categories")
async def list_categories(store_id: int, user: CurrentUser, db: DB):
    """The categories in use, most used first, for the category picker and the list filter."""
    await get_store_for(db, user, store_id)
    return await categories.store_categories(db, store_id)


@router.post("/stores/{store_id}/products", response_model=ProductOut)
async def create_product(store_id: int, body: ProductIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    await check_limit(db, user, "products")
    _check_video(store_id, body.video_url)
    data = body.model_dump(exclude={"images"})
    data["category"] = categories.clean(data.get("category"))
    p = Product(store_id=store_id, source="manual", created_by="user", **data)
    db.add(p)
    await db.flush()
    await product_code.apply_default_code(db, p)
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
    if "video_url" in data:
        _check_video(p.store_id, data["video_url"])
        data["video_url"] = data["video_url"] or ""
    if "category" in data:
        data["category"] = categories.clean(data["category"])
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


@router.post("/stores/{store_id}/uploads/video")
async def upload_video(store_id: int, user: CurrentUser, db: DB, file: UploadFile = File(...)):
    """One product video (for Instagram reels and video stories); checked with ffprobe, stored as is."""
    await get_store_for(db, user, store_id)
    ext = VIDEO_TYPES.get(file.content_type or "")
    if not ext:
        raise HTTPException(422, "فقط ویدیوی MP4، MOV یا WEBM مجاز است.")
    data = await file.read()
    if len(data) > video.MAX_VIDEO_BYTES:
        raise HTTPException(422, "حجم ویدیو حداکثر ۵۰ مگابایت است.")
    try:
        info = await video.probe_bytes(data, ext)
    except PlatformError as exc:
        raise HTTPException(422, exc.message_fa)
    if info.duration > video.MAX_SECONDS:
        raise HTTPException(422, "ویدیو حداکثر ۳ دقیقه می‌تواند باشد.")
    url = put(f"{store_id}/{uuid.uuid4().hex}{ext}", data, file.content_type or "")
    return {"url": url, "duration": round(info.duration, 1), "width": info.width, "height": info.height}


def _check_video(store_id: int, url: str | None) -> None:
    if url and not url.startswith(f"/media/{store_id}/"):
        raise HTTPException(422, "ویدیو باید از همین فروشگاه آپلود شده باشد.")


class PublishIn(BaseModel):
    channel_ids: list[int]
    show_price: bool | None = None  # write the price in the post; None = the store setting
    format: Literal["post", "story", "reel"] = "post"  # story and reel: Instagram only


def _check_format(ch: Channel, fmt: str) -> None:
    if fmt != "post" and ch.platform != "instagram":
        raise HTTPException(422, "استوری و ریلز فقط برای اینستاگرام است.")


@router.post("/products/{product_id}/publish", response_model=list[OperationOut], status_code=202)
async def publish(product_id: int, body: PublishIn, user: CurrentUser, db: DB):
    p = await _get(db, user, product_id)
    op_ids = []
    for cid in body.channel_ids:
        ch = await db.get(Channel, cid)
        if not ch or ch.store_id != p.store_id:
            raise HTTPException(404, "کانال پیدا نشد.")
        _check_format(ch, body.format)
        op_id = await queue_publish(db, p, ch, trigger="user", actor="user", show_price=body.show_price,
                                    format=body.format)
        if op_id:
            op_ids.append(op_id)
    await db.commit()
    return (await db.execute(select(Operation).where(Operation.id.in_(op_ids)))).scalars().all()


class PriceEdit(BaseModel):
    mode: Literal["set", "percent", "amount"]  # new price | ±percent | ±Toman
    value: float
    round_to: int = Field(0, ge=0)  # e.g. 1000: round the result to the nearest thousand Toman


class StockEdit(BaseModel):
    mode: Literal["set", "add"]  # "add" with a negative value subtracts
    value: int


class AttributeEdit(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    value: str = ""
    remove: bool = False


class BulkEdit(BaseModel):
    price: PriceEdit | None = None
    stock: StockEdit | None = None
    category: str | None = Field(None, max_length=120)  # "" clears it
    attribute: AttributeEdit | None = None


class BulkIn(BaseModel):
    product_ids: list[int] = []
    # Instead of ids: every product matching the list's filters (the "select all N products" checkbox).
    filters: ProductFilters | None = None
    action: str  # publish | tag | untag | archive | status | edit
    channel_ids: list[int] = []
    tag: str = ""
    status: str = ""
    show_price: bool | None = None
    format: Literal["post", "story", "reel"] = "post"
    edit: BulkEdit | None = None


BULK_MAX = 5000


def _new_price(price: int, e: PriceEdit) -> int:
    if e.mode == "set":
        new = e.value
    elif e.mode == "percent":
        new = price * (1 + e.value / 100)
    else:
        new = price + e.value
    if e.round_to:
        new = round(new / e.round_to) * e.round_to
    return max(0, int(round(new)))


def _apply_edit(p: Product, e: BulkEdit) -> bool:
    """Applies the manual bulk edit to one product; True when something changed."""
    before = (p.price, p.stock_qty, p.category, dict(p.attributes or {}))
    if e.price:
        p.price = _new_price(p.price, e.price)
    if e.stock:
        p.stock_qty = max(0, e.stock.value if e.stock.mode == "set" else p.stock_qty + e.stock.value)
    if e.category is not None:
        p.category = categories.clean(e.category)
    if e.attribute:
        name = normalize_fa(e.attribute.name).strip()
        attrs = dict(p.attributes or {})  # a new dict, so the JSON column is marked dirty
        if e.attribute.remove:
            attrs.pop(name, None)
        else:
            attrs[name] = normalize_fa(e.attribute.value).strip()
        p.attributes = attrs
    return (p.price, p.stock_qty, p.category, dict(p.attributes or {})) != before


@router.post("/stores/{store_id}/products/bulk")
async def bulk(store_id: int, body: BulkIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    if body.filters is not None:
        stmt = _filtered(store_id, body.filters)
    else:
        stmt = select(Product).where(Product.store_id == store_id, Product.id.in_(body.product_ids))
    if body.action == "edit":
        e = body.edit
        if not e or (e.price is None and e.stock is None and e.category is None and e.attribute is None):
            raise HTTPException(422, "هیچ تغییری انتخاب نشده.")
        if e.price and e.price.mode == "set" and e.price.value < 0:
            raise HTTPException(422, "قیمت نمی‌تواند منفی باشد.")
        if e.stock and e.stock.mode == "set" and e.stock.value < 0:
            raise HTTPException(422, "موجودی نمی‌تواند منفی باشد.")
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    if total > BULK_MAX:
        raise HTTPException(422, f"در هر بار حداکثر {BULK_MAX} محصول را می‌شود تغییر داد؛ با فیلتر محدودترش کن.")
    channels: list[Channel] = []
    if body.action == "publish":
        for cid in body.channel_ids:
            ch = await db.get(Channel, cid)
            if ch and ch.store_id == store_id:
                _check_format(ch, body.format)
                channels.append(ch)
    products = (await db.execute(stmt.options(selectinload(Product.images), selectinload(Product.listings)))).scalars().all()
    queued = changed = 0
    op_ids: list[int] = []
    for p in products:
        if body.action == "publish":
            for ch in channels:
                if op_id := await queue_publish(db, p, ch, trigger="user", actor="user", show_price=body.show_price,
                                                format=body.format):
                    queued += 1
                    op_ids.append(op_id)
        elif body.action == "tag" and body.tag:
            p.tags = sorted(set(p.tags or []) | {body.tag})
        elif body.action == "untag":
            p.tags = [t for t in p.tags or [] if t != body.tag]
        elif body.action == "archive":
            p.status = "archived"
        elif body.action == "status" and body.status in ("draft", "ready"):
            p.status = body.status
        elif body.action == "edit" and _apply_edit(p, body.edit):
            changed += 1
            for l in p.listings:  # same as a single edit: published posts now differ from the product
                if l.status == "published":
                    l.status = "out_of_sync"
    await db.commit()
    paced = []  # accounts that publish on a pace: when their last queued publish should go out
    for ch in channels:
        last = await db.scalar(select(func.max(Operation.run_after)).where(Operation.id.in_(op_ids or [0]),
                                                                           Operation.channel_id == ch.id))
        if last:
            n = await db.scalar(select(func.count()).select_from(Operation).where(Operation.id.in_(op_ids),
                                                                                Operation.channel_id == ch.id))
            paced.append({"channel_id": ch.id, "name": ch.display_name or ch.platform, "count": n, "last_at": last})
    return {"ok": True, "affected": len(products), "queued": queued, "changed": changed, "paced": paced}


@router.get("/stores/{store_id}/products.csv")
async def export_csv(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(Product).where(Product.store_id == store_id, Product.status != "archived"))).scalars()
    buf = io.StringIO()
    buf.write("﻿")
    w = csv.writer(buf)
    w.writerow(["id", "sku", "title", "category", "price", "stock", "source", "status", "tags"])
    for p in rows:
        w.writerow([p.id, p.sku, p.title, p.category or "", p.price, p.stock_qty, p.source, p.status, " ".join(p.tags or [])])
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
    category: str | None = None
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
            if "category" in fields:
                fields["category"] = categories.clean(fields["category"])
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
                await product_code.apply_default_code(db, p)
                known[p.sku] = p.id
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
