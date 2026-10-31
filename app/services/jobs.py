"""Operation handlers the worker runs: connect, health_check, publish, crawl, ai_product."""

import json
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.db import utcnow
from app.core.security import encrypt_credentials
from app.models import (
    AiProductRequest, Channel, ChannelListing, CrawlItem, CrawlSource, Operation, Product, ProductImage,
    Store,
)
from app.platforms.base import CrawledItem, EDIT_PRODUCT, PlatformError
from app.platforms.registry import get_adapter
from app.services import ai
from app.services.media import read_local, save_bytes
from app.services.product_images import make_photo
from app.services.ops import OpContext, record_event
from app.services.text import normalize_fa, parse_caption, parse_number


async def _load_product(db, product_id: int) -> Product:
    q = select(Product).where(Product.id == product_id).options(selectinload(Product.images))
    return (await db.execute(q)).scalar_one()


# ---------- channels ----------

async def run_connect(ctx: OpContext) -> None:
    op, db = ctx.op, ctx.db
    channel = await db.get(Channel, op.channel_id)
    adapter = get_adapter(channel.platform)
    try:
        result = await adapter.connect(ctx, channel, op.payload.get("data", {}))
    except PlatformError as exc:
        channel.status = "error"
        channel.last_error = exc.message_fa
        raise
    channel.credentials_encrypted = encrypt_credentials(result["credentials"])
    channel.config = {**channel.config, **result["config"]}
    channel.display_name = result["display_name"]
    channel.external_account_id = result["external_account_id"]
    channel.status = "connected"
    channel.last_error = ""
    channel.last_health_at = utcnow()
    op.payload = {}  # never keep the raw token/code on the operation
    op.result = {"display_name": channel.display_name}


async def run_health(ctx: OpContext) -> None:
    channel = await ctx.db.get(Channel, ctx.op.channel_id)
    try:
        await get_adapter(channel.platform).health(ctx, channel)
    except PlatformError as exc:
        channel.status = "expired" if exc.code in ("expired", "invalid_token", "bot_not_admin") else "error"
        channel.last_error = exc.message_fa
        raise
    channel.status, channel.last_error, channel.last_health_at = "connected", "", utcnow()


# ---------- publish ----------

async def run_publish(ctx: OpContext) -> None:
    op, db = ctx.op, ctx.db
    listing = await db.get(ChannelListing, op.ref_id)
    channel = await db.get(Channel, listing.channel_id)
    store = await db.get(Store, op.store_id)
    product = await _load_product(db, listing.product_id)
    adapter = get_adapter(channel.platform)
    listing.status = "publishing"
    await db.commit()
    try:
        if channel.status != "connected":
            raise PlatformError(f"کانال {adapter.label_fa} وصل نیست.", "channel_disconnected",
                                {"type": "reconnect", "label": "اتصال مجدد"})
        async with ctx.step("validate", "بررسی محصول با قوانین پلتفرم",
                            request={"title": product.title, "images": len(product.images)}) as s:
            errors = adapter.validate(product)
            s.response = {"errors": errors}
            if errors:
                raise PlatformError(" ".join(errors), "validation", EDIT_PRODUCT)
        result = await adapter.publish(ctx, channel, product, store, existing_remote_id=listing.remote_id)
    except Exception as exc:
        listing.status = "failed"
        listing.last_error_fa = getattr(exc, "message_fa", "خطای داخلی سیستم.")
        await record_event(db, op.store_id, "publish_failed", operation_id=op.id, channel_id=channel.id)
        raise
    listing.status = "published"
    listing.remote_id, listing.remote_url = result.remote_id, result.remote_url
    listing.published_by = "ai" if op.actor == "ai" else "user"
    listing.published_at = listing.published_at or utcnow()
    listing.last_synced_at = utcnow()
    listing.last_error_fa = ""
    op.result = {"remote_id": result.remote_id, "remote_url": result.remote_url}
    await record_event(db, op.store_id, "publish_ok", operation_id=op.id, channel_id=channel.id)


# ---------- crawl ----------

async def _extract(items: list[CrawledItem], mode: str, store_id: int) -> tuple[dict[str, dict], list[str]]:
    """Regex first; LLM for posts the rules couldn't complete (platform-wrappers approach)."""
    out: dict[str, dict] = {}
    pending: list[CrawledItem] = []
    notes: list[str] = []
    for it in items:
        if it.title and not it.caption:  # marketplace rows are already structured
            out[it.external_id] = {"title": it.title, "description": it.description, "price": it.price,
                                   "quantity": it.quantity, "sku": it.sku, "brand": it.brand,
                                   "category_hint": it.category, "method": "structured"}
            continue
        parsed = parse_caption(it.caption)
        out[it.external_id] = {**parsed, "method": "regex"}
        if mode == "llm" or (mode == "auto" and not parsed["complete"]):
            pending.append(it)
    if pending and mode != "regex":
        if not await ai.ai_available("caption_extract"):
            notes.append("هوش مصنوعی تنظیم نشده؛ فقط استخراج قاعده‌محور انجام شد.")
        else:
            for i in range(0, len(pending), 20):
                batch = pending[i:i + 20]
                try:
                    res = await ai.generate_json(
                        task="caption_extract", system=ai.CAPTION_EXTRACT_SYSTEM, store_id=store_id,
                        prompt=json.dumps({"posts": [{"message_id": b.external_id, "caption": b.caption} for b in batch]},
                                          ensure_ascii=False))
                except ai.AIError as exc:
                    notes.append(exc.message_fa)
                    continue
                for row in res.get("posts", []):
                    mid = str(row.get("message_id"))
                    if mid in out and row.get("title"):
                        out[mid].update({
                            "title": normalize_fa(row["title"])[:300],
                            "description": normalize_fa(row.get("description", "")),
                            "price": parse_number(str(row.get("price", 0))) or out[mid]["price"],
                            "quantity": parse_number(str(row.get("quantity", 0))) or out[mid]["quantity"],
                            "attributes": row.get("attributes") or {}, "method": "llm"})
    return out, notes


async def ingest_items(ctx: OpContext, *, store: Store, items: list[CrawledItem], source_key: dict[str, Any],
                       mode: str, channel: Channel | None, source_id: int | None, to_inbox: bool) -> dict[str, int]:
    db, op = ctx.db, ctx.op
    counts = {"seen": len(items), "created": 0, "updated": 0, "duplicate": 0, "skipped": 0, "failed": 0, "inbox": 0}
    async with ctx.step("extract", f"استخراج دیتای {len(items)} آیتم", request={"mode": mode}) as s:
        extracted, notes = await _extract(items, mode, store.id)
        s.response = {"methods": {k: v.get("method") for k, v in extracted.items()}, "notes": notes}
        s.message_fa = " ".join(notes)

    async with ctx.step("save", "ساخت/به‌روزرسانی محصولات") as s:
        existing = {
            (p.source_ref or {}).get("external_id"): p
            for p in (await db.execute(select(Product).where(Product.store_id == store.id))).scalars()
            if (p.source_ref or {}).get("key") == source_key.get("key")
        }
        for it in items:
            data = extracted.get(it.external_id, {})
            row = CrawlItem(operation_id=op.id, source_id=source_id, external_id=it.external_id,
                            raw={"caption": it.caption, "url": it.url, "images": it.image_urls}, extracted=data)
            db.add(row)
            if not data.get("title"):
                row.result, row.error_fa = "failed", "عنوان محصول از متن پیدا نشد."
                counts["failed"] += 1
                continue
            if data.get("method") != "structured" and not data.get("price") and not it.image_urls:
                row.result, row.error_fa = "skipped", "این پست شبیه محصول نبود (نه قیمت دارد نه عکس)."
                counts["skipped"] += 1
                continue
            prod = existing.get(it.external_id)
            if prod:
                changed = False
                for f in ("price", "stock_qty"):
                    new = data.get("price" if f == "price" else "quantity") or 0
                    if new and getattr(prod, f) != new:
                        setattr(prod, f, new)
                        changed = True
                row.product_id = prod.id
                row.result = "updated" if changed else "duplicate"
                counts[row.result] += 1
                continue
            prod = Product(store_id=store.id, title=data["title"], description=data.get("description", ""),
                           price=data.get("price") or 0, stock_qty=data.get("quantity") or 0, sku=data.get("sku", ""),
                           brand=data.get("brand", ""), category_hint=data.get("category_hint", ""),
                           attributes=data.get("attributes") or {}, source="crawl", created_by="crawler",
                           status="draft" if to_inbox else "ready",
                           tags=[h.lstrip("#") for h in data.get("hashtags", [])][:8],
                           source_ref={**source_key, "external_id": it.external_id, "url": it.url})
            db.add(prod)
            await db.flush()
            for pos, url in enumerate(it.image_urls[:10]):
                try:
                    if url.startswith("botfile:") and channel:
                        blob, name = await get_adapter(channel.platform).download_file(channel, url.removeprefix("botfile:"))
                        url = save_bytes(store.id, blob, name)
                    db.add(ProductImage(product_id=prod.id, url=url, position=pos, origin_url=url))
                except Exception:  # noqa: BLE001 - a missing photo shouldn't drop the product
                    row.error_fa = "بعضی عکس‌ها دریافت نشدند."
            row.product_id = prod.id
            row.result = "inbox" if to_inbox else "created"
            counts[row.result] += 1
            existing[it.external_id] = prod
        s.response = counts
    await record_event(db, store.id, "crawl_run", operation_id=op.id, channel_id=channel.id if channel else None)
    await record_event(db, store.id, "crawl_item", counts["created"] + counts["updated"] + counts["inbox"],
                       operation_id=op.id)
    await record_event(db, store.id, "product_created", counts["created"] + counts["inbox"], operation_id=op.id)
    return counts


async def run_crawl(ctx: OpContext) -> None:
    db, op = ctx.db, ctx.op
    source = await db.get(CrawlSource, op.ref_id)
    store = await db.get(Store, op.store_id)
    channel = await db.get(Channel, source.channel_id) if source.channel_id else None
    if source.type == "url":
        async with ctx.step("fetch_url", "دریافت صفحه", request={"url": source.url}) as s:
            page = await fetch_product_page(source.url)
            s.response = {k: v for k, v in page.items() if k != "text"}
        items = [CrawledItem(external_id=source.url, title=page.get("title", ""), description=page.get("description", ""),
                             price=page.get("price", 0), brand=page.get("brand", ""), image_urls=page.get("images", [])[:5],
                             url=source.url, caption="" if page.get("title") else page.get("text", "")[:3000])]
    else:
        if not channel or channel.status != "connected":
            raise PlatformError("کانال منبع کرال وصل نیست.", "channel_disconnected", {"type": "reconnect", "label": "اتصال مجدد"})
        items = await get_adapter(channel.platform).fetch_items(ctx, channel)
    counts = await ingest_items(ctx, store=store, items=items, mode=source.extraction_mode, channel=channel,
                                source_key={"key": f"src:{source.id}", "source_id": source.id,
                                            "platform": channel.platform if channel else "url"},
                                source_id=source.id, to_inbox=(store.settings or {}).get("crawl_to_inbox", False))
    source.last_run_at = utcnow()
    op.result = counts
    if counts["failed"] and counts["failed"] < counts["seen"]:
        op.status = "partial"
        op.error_message_fa = f"{counts['failed']} آیتم از {counts['seen']} آیتم استخراج نشد."
    if counts["seen"] == 0:
        op.result["note"] = ctx.empty_reason or "آیتم جدیدی پیدا نشد."


# ---------- link fetching (url crawl) ----------

async def fetch_product_page(url: str) -> dict[str, Any]:
    """JSON-LD Product first, then OpenGraph/meta, then visible text for the LLM."""
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 SelloBot/1.0", "Accept-Language": "fa,en"}) as c:
            resp = await c.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise PlatformError("صفحهٔ لینک در دسترس نبود.", "url_unreachable", detail=str(exc)) from exc
    soup = BeautifulSoup(resp.text, "html.parser")
    out: dict[str, Any] = {"url": url, "method": "html"}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for node in nodes:
            if isinstance(node, dict) and "Product" in str(node.get("@type")):
                offers = node.get("offers") or {}
                offers = offers[0] if isinstance(offers, list) and offers else offers
                imgs = node.get("image") or []
                out.update({
                    "method": "json-ld", "title": node.get("name", ""), "description": node.get("description", ""),
                    "brand": (node.get("brand") or {}).get("name", "") if isinstance(node.get("brand"), dict) else str(node.get("brand") or ""),
                    "price": parse_number(str(offers.get("price", 0))) if isinstance(offers, dict) else 0,
                    "images": imgs if isinstance(imgs, list) else [imgs],
                    "attributes": {p.get("name"): p.get("value") for p in node.get("additionalProperty", []) if isinstance(p, dict)},
                })
                break
    def meta(prop: str) -> str:
        t = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return (t.get("content") or "").strip() if t else ""
    out.setdefault("title", meta("og:title") or (soup.title.string.strip() if soup.title and soup.title.string else ""))
    out.setdefault("description", meta("og:description") or meta("description"))
    out.setdefault("images", [meta("og:image")] if meta("og:image") else [])
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    out["text"] = re.sub(r"\s+", " ", soup.get_text(" "))[:6000]
    return out


# ---------- AI: product from the seller's photos and description ----------

async def run_ai_product(ctx: OpContext) -> None:
    req = await ctx.db.get(AiProductRequest, ctx.op.ref_id)
    await generate_product(ctx, req)


async def generate_product(ctx: OpContext, ref: AiProductRequest) -> Product:
    db, op = ctx.db, ctx.op
    store = await db.get(Store, op.store_id)
    images = [b for b in (read_local(u) for u in ref.images) if b][:4]
    photo_role = ("The attached photo is a DESIGN/artwork the seller prints on the product, not the product itself; "
                  "describe the product from the seller's note, with this design on it."
                  if (op.payload or {}).get("photo_kind") == "design" else "")
    prompt = json.dumps({"seller_note": ref.note, "photo_role": photo_role, "price": ref.price}, ensure_ascii=False)
    async with ctx.step("generate", "ساخت محصول با هوش مصنوعی", request={"images": len(images)}) as s:
        result = await ai.generate_json(task="product_generate", system=ai.PRODUCT_FROM_PHOTO_SYSTEM,
                                        prompt=prompt, images=images, store_id=store.id)
        s.response = result
    product = Product(store_id=store.id, title=normalize_fa(result.get("title") or "محصول جدید")[:300],
                      description=normalize_fa(result.get("description", "")), price=ref.price or int(result.get("price") or 0),
                      stock_qty=ref.stock_qty or parse_number(str(result.get("quantity") or 0)), brand=result.get("brand", ""), category_hint=result.get("category_hint", ""),
                      attributes=result.get("attributes") or {}, tags=(result.get("tags") or [])[:8], source="ai",
                      created_by="ai", status="draft",
                      source_ref={"ai_request_id": ref.id, "field_sources": result.get("field_sources", {})})
    db.add(product)
    await db.flush()
    for pos, url in enumerate(ref.images[:10]):
        db.add(ProductImage(product_id=product.id, url=url, position=pos))
    ref.result_product_ids = [product.id]
    op.product_id = product.id
    op.result = {"product_id": product.id}
    await record_event(db, store.id, "product_created", operation_id=op.id)
    problems = []
    if (op.payload or {}).get("clean_image"):
        await db.commit()  # keep the product even if the image step fails
        source = ref.images[0] if ref.images else None
        try:
            url = await make_photo(ctx, store_id=store.id, image_url=source, title=product.title,
                                   note=ref.note or product.description[:400],
                                   category=str(result.get("image_category") or "general"),
                                   design=op.payload.get("photo_kind") == "design")
            await _set_main_image(db, product.id, url, origin=source or "")
            op.result = {**op.result, "clean_image": url}
        except (ai.AIError, ValueError) as exc:
            # Sellers get a plain sentence; the technical reason stays in the step log.
            problems.append("محصول ساخته شد ولی عکس ساخته نشد؛ با یک عکس دیگر دوباره امتحان کن."
                            if isinstance(exc, ai.AIError) and exc.code == "ai_no_image"
                            else f"محصول ساخته شد ولی عکس ساخته نشد: {getattr(exc, 'message_fa', None) or exc}")
    if problems:
        op.status = "partial"
        op.error_message_fa = " ".join(problems)
    return product


async def _set_main_image(db, product_id: int, url: str, origin: str) -> None:
    """Puts the cleaned photo first and keeps the seller's originals after it."""
    for img in (await db.execute(select(ProductImage).where(ProductImage.product_id == product_id))).scalars():
        img.position += 1
    db.add(ProductImage(product_id=product_id, url=url, position=0, origin_url=origin))
    await db.flush()


HANDLERS = {
    "connect": run_connect,
    "health_check": run_health,
    "publish": run_publish,
    "crawl": run_crawl,
    "ai_product": run_ai_product,
}
