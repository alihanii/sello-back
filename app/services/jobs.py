"""Operation handlers the worker runs: connect, health_check, publish, crawl, reference, style."""

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
    Approval, Automation, Channel, ChannelListing, CrawlItem, CrawlSource, Operation, Product, ProductImage, ReferenceSet,
    Store, StoreStyleProfile,
)
from app.platforms.base import CrawledItem, EDIT_PRODUCT, PlatformError
from app.platforms.registry import get_adapter
from app.services import ai
from app.services.media import read_local, save_bytes
from app.services.product_images import make_photo
from app.services.automation import queue_publish
from app.services.notify import notify
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


# ---------- link fetching (reference + url crawl) ----------

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


# ---------- AI: reference-based generation & style profile ----------

async def run_reference(ctx: OpContext) -> None:
    ref = await ctx.db.get(ReferenceSet, ctx.op.ref_id)
    await generate_from_reference(ctx, ref)


async def generate_from_reference(ctx: OpContext, ref: ReferenceSet) -> Product:
    db, op = ctx.db, ctx.op
    store = await db.get(Store, op.store_id)
    pages: list[dict] = []
    fetches = []
    for i, url in enumerate(ref.urls):
        try:
            async with ctx.step(f"link_{i + 1}", f"کرال لینک {i + 1} از {len(ref.urls)}", request={"url": url}) as s:
                page = await fetch_product_page(url)
                s.response = {k: v for k, v in page.items() if k != "text"}
            pages.append({k: v for k, v in page.items() if k != "images"})
            fetches.append({"url": url, "status": "ok", "method": page["method"]})
        except PlatformError as exc:
            fetches.append({"url": url, "status": "failed", "error_fa": exc.message_fa})
    ref.url_fetches = fetches

    samples, profile = [], None
    if ref.use_style_profile:
        async with ctx.step("style", "خواندن سبک فروشگاه") as s:
            prof = (await db.execute(select(StoreStyleProfile).where(StoreStyleProfile.store_id == store.id))).scalar_one_or_none()
            profile = prof.profile if prof else None
            ids = ref.product_ids or []
            q = select(Product).where(Product.store_id == store.id, Product.source != "ai")
            q = q.where(Product.id.in_(ids)) if ids else q.order_by(Product.is_golden_sample.desc(), Product.id.desc()).limit(5)
            samples = [{"title": p.title, "description": p.description[:600], "price": p.price}
                       for p in (await db.execute(q)).scalars()]
            s.response = {"has_profile": bool(profile), "samples": len(samples)}

    images = [b for b in (read_local(u) for u in ref.images) if b][:4]
    photo_role = ("The attached photo is a DESIGN/artwork the seller prints on the product, not the product itself; "
                  "describe the product from the seller's note, with this design on it."
                  if (op.payload or {}).get("photo_kind") == "design" else "")
    prompt = json.dumps({"seller_note": ref.note, "photo_role": photo_role, "price": ref.price, "style_profile": profile,
                         "shop_examples": samples, "reference_pages": pages}, ensure_ascii=False)
    async with ctx.step("generate", "ساخت محصول با هوش مصنوعی", request={"images": len(images), "links": len(pages)}) as s:
        result = await ai.generate_json(task="product_generate", system=ai.PRODUCT_FROM_REFERENCE_SYSTEM,
                                        prompt=prompt, images=images, store_id=store.id)
        s.response = result
    product = Product(store_id=store.id, title=normalize_fa(result.get("title") or "محصول جدید")[:300],
                      description=normalize_fa(result.get("description", "")), price=ref.price or int(result.get("price") or 0),
                      stock_qty=ref.stock_qty or parse_number(str(result.get("quantity") or 0)), brand=result.get("brand", ""), category_hint=result.get("category_hint", ""),
                      attributes=result.get("attributes") or {}, tags=(result.get("tags") or [])[:8], source="ai",
                      created_by="ai", status="draft",
                      source_ref={"reference_set_id": ref.id, "urls": ref.urls, "field_sources": result.get("field_sources", {})})
    db.add(product)
    await db.flush()
    for pos, url in enumerate(ref.images[:10]):
        db.add(ProductImage(product_id=product.id, url=url, position=pos))
    ref.result_product_ids = [product.id]
    op.product_id = product.id
    op.result = {"product_id": product.id, "url_fetches": fetches}
    await record_event(db, store.id, "product_created", operation_id=op.id)
    problems = []
    if any(f["status"] == "failed" for f in fetches):
        problems.append("بعضی لینک‌ها کرال نشدند؛ محصول با بقیهٔ رفرنس‌ها ساخته شد.")
    if (op.payload or {}).get("clean_image"):
        await db.commit()  # keep the product even if the image step fails
        source = ref.images[0] if ref.images else None
        try:
            url = await make_photo(ctx, store_id=store.id, image_url=source, title=product.title,
                                   note=ref.note or product.description[:400], product_id=product.id,
                                   category=str(result.get("image_category") or "general"),
                                   style_urls=op.payload.get("style_images") or [],
                                   style_product_ids=op.payload.get("style_product_ids") or [],
                                   design=op.payload.get("photo_kind") == "design")
            await _set_main_image(db, product.id, url, origin=source or "")
            op.result = {**op.result, "clean_image": url}
        except (ai.AIError, ValueError) as exc:
            what = "تمیز" if source else "ساخته"
            problems.append(f"محصول ساخته شد ولی عکس {what} نشد: {getattr(exc, 'message_fa', None) or exc}")
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


async def run_image_clean(ctx: OpContext) -> None:
    """Cleans one existing product photo (product page button)."""
    db, op = ctx.db, ctx.op
    product = await _load_product(db, op.product_id)
    image = await db.get(ProductImage, int(op.payload["image_id"])) if op.payload.get("image_id") else None
    if op.payload.get("image_id") and (not image or image.product_id != product.id):
        raise PlatformError("عکس پیدا نشد.", "image_not_found")
    try:
        url = await make_photo(ctx, store_id=product.store_id, image_url=image.url if image else None, title=product.title,
                               note=product.description[:500], product_id=product.id,
                               category=str(op.payload.get("category") or "general"),
                               style_urls=op.payload.get("style_images") or [],
                               style_product_ids=op.payload.get("style_product_ids") or [])
    except ValueError as exc:
        raise PlatformError(str(exc), "image_not_found")
    await _set_main_image(db, product.id, url, origin=image.url if image else "")
    op.result = {"clean_image": url, "source_image_id": image.id if image else None}


async def _style_context(db, store_id: int) -> tuple[dict | None, list[dict]]:
    prof = (await db.execute(select(StoreStyleProfile).where(StoreStyleProfile.store_id == store_id))).scalar_one_or_none()
    q = (select(Product).where(Product.store_id == store_id, Product.source != "ai")
         .order_by(Product.is_golden_sample.desc(), Product.id.desc()).limit(4))
    samples = [{"title": p.title, "description": p.description[:500]} for p in (await db.execute(q)).scalars()]
    return (prof.profile if prof else None), samples


async def _pick_catalog_product(db, rule: Automation) -> Product | None:
    """Oldest product with photos that isn't yet on every target channel and isn't already waiting for approval."""
    targets = set((rule.actions or {}).get("channel_ids", []))
    pending = select(Approval.product_id).where(Approval.store_id == rule.store_id, Approval.status == "pending")
    q = (select(Product).where(Product.store_id == rule.store_id, Product.status.in_(["ready", "draft"]),
                               Product.id.not_in(pending))
         .options(selectinload(Product.images), selectinload(Product.listings)).order_by(Product.id))
    for p in (await db.execute(q)).scalars():
        done = {l.channel_id for l in p.listings if l.status in ("published", "queued", "publishing")}
        if p.images and targets - done:
            return p
    return None


async def run_ai_post(ctx: OpContext) -> None:
    """Content automation: pick (or build) a product, have AI write the post, then ask the owner to approve."""
    db, op = ctx.db, ctx.op
    rule = await db.get(Automation, op.ref_id)
    if not rule:
        raise PlatformError("اتوماسیون این کار حذف شده است.", "automation_missing")
    a = rule.actions or {}
    source = a.get("source", "catalog")

    if source == "references":
        queue = list(a.get("queue") or [])
        if not queue:
            raise PlatformError("صف رفرنس‌های این اتوماسیون خالی است؛ عکس یا لینک جدید اضافه کنید.", "queue_empty",
                                {"type": "edit_automation", "label": "افزودن رفرنس"})
        item, rule.actions = queue[0], {**a, "queue": queue[1:]}
        ref = ReferenceSet(store_id=rule.store_id, images=item.get("images", []), urls=item.get("urls", []),
                           note="\n".join(x for x in (a.get("brief", ""), item.get("note", "")) if x),
                           price=int(item.get("price") or 0), stock_qty=int(item.get("stock_qty") or 0),
                           use_style_profile=a.get("use_style_profile", True), operation_id=op.id)
        db.add(ref)
        await db.flush()
        product = await generate_from_reference(ctx, ref)
        await db.refresh(product, ["images"])
    else:
        async with ctx.step("pick", "انتخاب محصول بعدی از کاتالوگ") as s:
            product = await _pick_catalog_product(db, rule)
            if not product:
                raise PlatformError("محصول عکس‌داری که هنوز در این کانال‌ها منتشر نشده باشد پیدا نشد.", "nothing_to_post",
                                    {"type": "edit_product", "label": "افزودن محصول"})
            s.response = {"product_id": product.id, "title": product.title}
        op.product_id = product.id

    profile, samples = await _style_context(db, rule.store_id)
    recent = [r for (r,) in (await db.execute(select(Product.post_caption).where(
        Product.store_id == rule.store_id, Product.post_caption != "").order_by(Product.updated_at.desc()).limit(5)))]
    images = [b for b in (read_local(i.url) for i in product.images[:3]) if b]
    prompt = json.dumps({"product": {"title": product.title, "description": product.description[:1500],
                                     "attributes": product.attributes, "category": product.category_hint},
                         "brief": a.get("brief", ""), "tone": a.get("tone", ""), "style_profile": profile,
                         "example_posts": samples, "recent_ai_posts": [r[:300] for r in recent]}, ensure_ascii=False)
    async with ctx.step("write", "نوشتن متن پست با هوش مصنوعی", request={"product_id": product.id, "images": len(images)}) as s:
        result = await ai.generate_json(task="caption_rewrite", system=ai.POST_WRITER_SYSTEM, prompt=prompt,
                                        images=images, store_id=rule.store_id)
        s.response = result
    caption = normalize_fa(result.get("caption", "")).strip()
    if not caption:
        raise PlatformError("هوش مصنوعی متن پست را برنگرداند؛ دوباره تلاش کنید.", "empty_caption", {"type": "retry", "label": "تلاش دوباره"})
    product.post_caption = caption
    if product.status == "draft":
        product.status = "ready"

    mode = a.get("mode", "approval")
    channel_ids = a.get("channel_ids", [])
    if mode == "auto":
        for cid in channel_ids:
            ch = await db.get(Channel, cid)
            if ch and ch.status == "connected":
                await queue_publish(db, product, ch, trigger="automation", actor="ai")
        op.result = {"product_id": product.id, "queued": True}
        return
    approval = Approval(store_id=rule.store_id, product_id=product.id, automation_id=rule.id, channel_ids=channel_ids,
                        kind="ai_post", operation_id=op.id)
    db.add(approval)
    await db.flush()
    await notify(db, rule.store_id, "approval", f"پست جدید آمادهٔ تأیید: {product.title[:80]}", caption[:240],
                 link=f"/app/s/{rule.store_id}/automations?approval={approval.id}",
                 image_url=product.images[0].url if product.images else "")
    op.result = {"product_id": product.id, "approval_id": approval.id}


async def run_style_profile(ctx: OpContext) -> None:
    db, op = ctx.db, ctx.op
    q = (select(Product).where(Product.store_id == op.store_id, Product.source != "ai")
         .order_by(Product.is_golden_sample.desc(), Product.id.desc()).limit(25))
    products = list((await db.execute(q)).scalars())
    if len(products) < 3:
        raise PlatformError("برای ساخت پروفایل سبک حداقل ۳ محصول (دستی یا کرال‌شده) لازم است.", "too_few_products")
    async with ctx.step("analyse", f"تحلیل {len(products)} محصول") as s:
        result = await ai.generate_json(
            task="style_profile", system=ai.STYLE_PROFILE_SYSTEM, store_id=op.store_id,
            prompt=json.dumps([{"title": p.title, "description": p.description[:800], "price": p.price,
                                "category": p.category_hint} for p in products], ensure_ascii=False))
        s.response = result
    prof = (await db.execute(select(StoreStyleProfile).where(StoreStyleProfile.store_id == op.store_id))).scalar_one_or_none()
    if not prof:
        prof = StoreStyleProfile(store_id=op.store_id)
        db.add(prof)
    prof.profile = result
    prof.sample_product_ids = [p.id for p in products]
    op.result = {"profile": result}


HANDLERS = {
    "ai_post": run_ai_post,
    "connect": run_connect,
    "health_check": run_health,
    "publish": run_publish,
    "crawl": run_crawl,
    "image_clean": run_image_clean,
    "reference": run_reference,
    "style_profile": run_style_profile,
}
