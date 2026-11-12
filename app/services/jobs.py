"""Operation handlers the worker runs: connect, health_check, publish, crawl, ai_product."""

import asyncio
import json
from datetime import timedelta
import mimetypes
from collections.abc import Awaitable, Callable
from typing import Any

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
from app.services import ai, categories, product_code
from app.services.images import to_jpeg
from app.services.media import read_local, save_bytes
from app.services import instagram_crawl as ig
from app.services.integrations import instagram_crawler
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

async def _paced(ctx: OpContext, channel: Channel) -> bool:
    """Instagram app-API accounts and LinkedIn accounts publish on a pace (random gap, daily cap, pauses). True when this publish must
    wait: the operation goes back to the queue with `run_after` set, and nothing else changes."""
    from app.platforms import instagram_private as igp
    from app.services.integrations import instagram_app, linkedin_app

    mode = (channel.config or {}).get("mode")
    if channel.platform == "instagram" and mode == "private":
        get_app = instagram_app
    elif channel.platform == "linkedin":
        get_app = linkedin_app  # same pacing rules, LinkedIn's own daily cap and gaps
    else:
        return False
    op, db = ctx.op, ctx.db
    app = await get_app(db)
    lock = igp._locks.setdefault(channel.id, asyncio.Lock())
    async with lock:
        await db.refresh(channel)
        now = utcnow()
        when = igp.next_slot(channel, app, now)
        if when is None:
            igp.reserve(channel, app, now)
            await db.commit()
            return False
    op.status, op.started_at, op.run_after = "queued", None, when
    local = when + timedelta(hours=3, minutes=30)
    reason = (channel.config or {}).get("pause_reason") if (channel.config or {}).get("paused_until") else ""
    from app.services.ops import turn_text

    op.current_step = turn_text(local, (now + timedelta(hours=3, minutes=30)).date(), app, bool(reason))
    await db.commit()
    return True


async def run_publish(ctx: OpContext) -> None:
    op, db = ctx.op, ctx.db
    fmt = (op.payload or {}).get("format") or "post"
    listing = await db.get(ChannelListing, op.ref_id) if op.ref_type == "listing" else None
    channel = await db.get(Channel, listing.channel_id if listing else op.channel_id)
    store = await db.get(Store, op.store_id)
    product = await _load_product(db, listing.product_id if listing else op.product_id)
    adapter = get_adapter(channel.platform)
    if channel.status == "connected" and await _paced(ctx, channel):
        return
    if listing:
        listing.status = "publishing"
    await db.commit()
    try:
        if channel.status != "connected":
            raise PlatformError(f"کانال {adapter.label_fa} وصل نیست.", "channel_disconnected",
                                {"type": "reconnect", "label": "اتصال مجدد"})
        async with ctx.step("validate", "بررسی محصول با قوانین پلتفرم",
                            request={"title": product.title, "images": len(product.images), "format": fmt}) as s:
            if hasattr(adapter, "validate_publish"):
                errors = adapter.validate_publish(channel, product, fmt)
            elif fmt != "post":
                errors = [f"{adapter.label_fa} فقط پست معمولی دارد؛ استوری و ریلز مخصوص اینستاگرام است."]
            else:
                errors = adapter.validate(product)
            s.response = {"errors": errors}
            if errors:
                raise PlatformError(" ".join(errors), "validation", EDIT_PRODUCT)
        result = await adapter.publish(ctx, channel, product, store,
                                       existing_remote_id=listing.remote_id if listing else "")
    except Exception as exc:
        if listing:
            listing.status = "failed"
            listing.last_error_fa = getattr(exc, "message_fa", "خطای داخلی سیستم.")
        await record_event(db, op.store_id, "publish_failed", operation_id=op.id, channel_id=channel.id)
        raise
    if listing:
        listing.status = "published"
        listing.remote_id, listing.remote_url = result.remote_id, result.remote_url
        listing.format = fmt
        listing.published_by = "ai" if op.actor == "ai" else "user"
        listing.published_at = listing.published_at or utcnow()
        listing.last_synced_at = utcnow()
        listing.last_error_fa = ""
    op.result = {"remote_id": result.remote_id, "remote_url": result.remote_url, "format": fmt}
    await record_event(db, op.store_id, "publish_ok", operation_id=op.id, channel_id=channel.id)


# ---------- crawl ----------

ImageLoader = Callable[[CrawledItem], Awaitable[list[tuple[bytes, str]]]]
# "AI only" mode runs once per post, so it is kept cheap: the first photo only, small and read at low resolution,
# a short answer with no description and no thinking.
AI_ONLY_PHOTOS = 1
AI_ONLY_PHOTO_SIDE = 512
AI_ONLY_MAX_TOKENS = 800
AI_ONLY_PARALLEL = 4
# Thinking is billed as output and these are extraction/copywriting tasks: a cap turns it off on Gemini Flash.
AUTO_BATCH_TOKENS_PER_POST = 600
PRODUCT_MAX_TOKENS = 2048


def clean_attributes(raw: Any) -> dict[str, str]:
    """AI attributes as {name: value} strings. Accepts a dict or a list of {name, value} pairs; drops empty values."""
    if isinstance(raw, list):
        raw = {str(r.get("name") or r.get("key") or ""): r.get("value") for r in raw if isinstance(r, dict)}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            v = "، ".join(str(x).strip() for x in v if x not in (None, ""))
        name, value = normalize_fa(str(k)).strip(), normalize_fa("" if v is None else str(v)).strip()
        if name and value:
            out[name] = value
    return out


def _apply_ai_row(out: dict[str, dict], row: dict, mode: str, shop_categories: list[str]) -> None:
    mid = str(row.get("message_id"))
    if mid not in out or not row.get("title"):
        return
    title = normalize_fa(row["title"])[:300]
    description = row.get("description")
    if description is None:  # AI only doesn't rewrite the text: keep the caption without its price/stock lines
        first = out[mid].get("title", "")
        description = "\n".join(x for x in (first if first != title else "", out[mid].get("description", "")) if x)
    out[mid].update({
        "title": title,
        "description": normalize_fa(description),
        "price": parse_number(str(row.get("price", 0))) or out[mid]["price"],
        "quantity": parse_number(str(row.get("quantity", 0))) or out[mid]["quantity"],
        "attributes": clean_attributes(row.get("attributes")) or out[mid].get("attributes") or {},
        "method": "llm"})
    if mode == "llm" and row.get("category_hint"):
        out[mid]["category_hint"] = normalize_fa(str(row["category_hint"]))[:200]
    if mode == "llm" and row.get("category"):
        category = categories.match(row["category"], shop_categories)
        out[mid]["category"] = category
        if category not in shop_categories:  # later posts in this crawl reuse it
            shop_categories.append(category)


async def _extract(items: list[CrawledItem], mode: str, store_id: int,
                   shop_categories: list[str] | None = None,
                   load_images: ImageLoader | None = None) -> tuple[dict[str, dict], list[str]]:
    """Regex first; LLM for posts the rules couldn't complete (platform-wrappers approach).
    "AI only" (llm) reads each post on its own with its photos, like AI product creation, so details that are only
    in the photo (colour, material, size label) reach the attributes too."""
    shop_categories = list(shop_categories or [])
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
    if not pending or mode == "regex":
        return out, notes
    if not await ai.ai_available("caption_extract"):
        notes.append("هوش مصنوعی تنظیم نشده؛ فقط استخراج قاعده‌محور انجام شد.")
        return out, notes
    if mode == "llm":
        gate = asyncio.Semaphore(AI_ONLY_PARALLEL)

        async def one(it: CrawledItem) -> None:
            async with gate:
                images: list[tuple[bytes, str]] = []
                if load_images and it.image_urls:
                    try:
                        images = (await load_images(it))[:AI_ONLY_PHOTOS]
                    except Exception:  # noqa: BLE001 - the caption alone still gives a product
                        images = []
                try:
                    res = await ai.generate_json(
                        task="caption_extract", store_id=store_id, system=ai.CAPTION_EXTRACT_FULL_SYSTEM,
                        images=images, low_res_images=True, max_tokens=AI_ONLY_MAX_TOKENS,
                        prompt=json.dumps({"posts": [{"message_id": it.external_id, "caption": it.caption}],
                                           "shop_categories": shop_categories[:categories.PROMPT_LIMIT]},
                                          ensure_ascii=False))
                except ai.AIError as exc:
                    notes.append(exc.message_fa)
                    return
                for row in res.get("posts", []):
                    _apply_ai_row(out, row, mode, shop_categories)

        await asyncio.gather(*(one(it) for it in pending))
        return out, list(dict.fromkeys(notes))
    for i in range(0, len(pending), 20):
        batch = pending[i:i + 20]
        try:
            res = await ai.generate_json(
                task="caption_extract", store_id=store_id, system=ai.CAPTION_EXTRACT_SYSTEM,
                max_tokens=AUTO_BATCH_TOKENS_PER_POST * len(batch),
                prompt=json.dumps({"posts": [{"message_id": b.external_id, "caption": b.caption} for b in batch]},
                                  ensure_ascii=False))
        except ai.AIError as exc:
            notes.append(exc.message_fa)
            continue
        for row in res.get("posts", []):
            _apply_ai_row(out, row, mode, shop_categories)
    return out, notes


async def ingest_items(ctx: OpContext, *, store: Store, items: list[CrawledItem], source_key: dict[str, Any],
                       mode: str, channel: Channel | None, source_id: int | None, to_inbox: bool,
                       fetch_image=None) -> dict[str, int]:
    """`fetch_image(url) -> (bytes, content_type)` copies remote photos (Instagram links expire) to our media."""
    db, op = ctx.db, ctx.op
    counts = {"seen": len(items), "created": 0, "updated": 0, "duplicate": 0, "skipped": 0, "failed": 0, "inbox": 0}
    fetched: dict[str, tuple[bytes, str, str]] = {}  # url -> (bytes, filename, content type); AI and save share one download

    async def fetch(url: str) -> tuple[bytes, str, str] | None:
        if url not in fetched:
            if url.startswith("botfile:") and channel:
                blob, name = await get_adapter(channel.platform).download_file(channel, url.removeprefix("botfile:"))
                fetched[url] = (blob, name, "")
            elif fetch_image and url.startswith("http"):
                blob, ctype = await fetch_image(url)
                fetched[url] = (blob, "", ctype or "")
            else:
                return None
        return fetched[url]

    async def load_images(it: CrawledItem) -> list[tuple[bytes, str]]:
        photos = []
        for url in it.image_urls[:AI_ONLY_PHOTOS]:
            try:
                got = await fetch(url)
            except Exception:  # noqa: BLE001, S112 - one missing photo shouldn't drop the others
                continue
            if got:
                try:
                    photos.append((to_jpeg(got[0], max_side=AI_ONLY_PHOTO_SIDE), "image/jpeg"))
                except PlatformError:
                    photos.append((got[0], got[2] or mimetypes.guess_type(got[1])[0] or "image/jpeg"))
        return photos

    async with ctx.step("extract", f"استخراج دیتای {len(items)} آیتم", request={"mode": mode}) as s:
        shop_categories = await categories.category_names(db, store.id) if mode == "llm" else []
        extracted, notes = await _extract(items, mode, store.id, shop_categories, load_images=load_images)
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
                           category=data.get("category", ""),
                           attributes=data.get("attributes") or {}, source="crawl", created_by="crawler",
                           status="draft" if to_inbox else "ready",
                           tags=[h.lstrip("#") for h in data.get("hashtags", [])][:8],
                           source_ref={**source_key, "external_id": it.external_id, "url": it.url})
            db.add(prod)
            await db.flush()
            await product_code.apply_default_code(db, prod, it.caption or "")
            for pos, url in enumerate(it.image_urls[:10]):
                try:
                    got = await fetch(url)
                    if got:
                        url = save_bytes(store.id, got[0], got[1], content_type=got[2])
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
    if source.type == "instagram_profile":
        await _crawl_instagram(ctx, source, store)
        return
    if source.type == "url":  # removed source type; old rows may still exist
        raise PlatformError("کرال با لینک حذف شده است.", "source_removed")
    if not channel or channel.status != "connected":
        raise PlatformError("کانال منبع کرال وصل نیست.", "channel_disconnected", {"type": "reconnect", "label": "اتصال مجدد"})
    items = await get_adapter(channel.platform).fetch_items(ctx, channel)
    counts = await ingest_items(ctx, store=store, items=items, mode=source.extraction_mode, channel=channel,
                                source_key={"key": f"src:{source.id}", "source_id": source.id,
                                            "platform": channel.platform},
                                source_id=source.id, to_inbox=(store.settings or {}).get("crawl_to_inbox", False))
    source.last_run_at = utcnow()
    op.result = counts
    if counts["failed"] and counts["failed"] < counts["seen"]:
        op.status = "partial"
        op.error_message_fa = f"{counts['failed']} آیتم از {counts['seen']} آیتم استخراج نشد."
    if counts["seen"] == 0:
        op.result["note"] = ctx.empty_reason or "آیتم جدیدی پیدا نشد."


async def _crawl_instagram(ctx: OpContext, source: CrawlSource, store: Store) -> None:
    """Newest 25 posts of a public page. Posts crawled before are never imported again (reported as not loaded),
    reels are skipped unless the source allows them, and then only their cover image is used."""
    db, op = ctx.db, ctx.op
    username, include_reels = source.url, bool((source.options or {}).get("include_reels"))
    cfg = await instagram_crawler(db)
    async with ctx.step("fetch_instagram", f"دریافت ۲۵ پست آخر ‎@{username}", request={"username": username,
                        "include_reels": include_reels, "provider": cfg["provider"]}) as s:
        posts = await ig.fetch_posts(cfg, username)
        s.response = {"posts": len(posts), "kinds": {k: sum(p.kind == k for p in posts) for k in ("image", "carousel", "reel")}}
    key = f"ig:{username}"  # per page, not per source: re-adding the same page doesn't re-import its posts
    known = {(p.source_ref or {}).get("external_id")
             for p in (await db.execute(select(Product).where(Product.store_id == store.id))).scalars()
             if (p.source_ref or {}).get("key") == key}
    fresh, reels, already = [], 0, 0
    for p in posts:
        if p.shortcode in known:
            already += 1
            db.add(CrawlItem(operation_id=op.id, source_id=source.id, external_id=p.shortcode, result="duplicate",
                             raw={"url": p.url, "kind": p.kind}, error_fa="قبلاً کرال شده؛ دوباره بارگذاری نشد."))
        elif p.kind == "reel" and not include_reels:
            reels += 1
            db.add(CrawlItem(operation_id=op.id, source_id=source.id, external_id=p.shortcode, result="skipped",
                             raw={"url": p.url, "kind": p.kind}, error_fa="ریلز است؛ کرال ریلز خاموش است."))
        elif not p.image_urls:
            db.add(CrawlItem(operation_id=op.id, source_id=source.id, external_id=p.shortcode, result="skipped",
                             raw={"url": p.url, "kind": p.kind}, error_fa="این پست عکس ندارد."))
        else:
            fresh.append(CrawledItem(external_id=p.shortcode, caption=p.caption, image_urls=p.image_urls[:1] if p.kind == "reel"
                                     else p.image_urls, url=p.url, raw={"kind": p.kind}))
    counts = await ingest_items(ctx, store=store, items=fresh, mode=source.extraction_mode, channel=None,
                                source_key={"key": key, "source_id": source.id, "platform": "instagram"},
                                source_id=source.id, to_inbox=(store.settings or {}).get("crawl_to_inbox", False),
                                fetch_image=lambda u: ig.download(u, cfg["proxy"]))
    skipped_other = len(posts) - len(fresh) - already - reels
    counts.update(seen=len(posts), not_loaded=already, reels_skipped=reels, skipped=counts["skipped"] + skipped_other)
    source.last_run_at = utcnow()
    op.result = counts
    notes = []
    if already:
        notes.append(f"{already} پست قبلاً کرال شده بود و دوباره بارگذاری نشد.")
    if reels:
        notes.append(f"{reels} ریلز رد شد؛ اگر ریلزها را هم می‌خواهی، گزینهٔ «ریلزها هم کرال شوند» را روشن کن (فقط عکس کاورشان برداشته می‌شود).")
    if not posts:
        notes.append("این پیج پستی نداشت.")
    if notes:
        op.result["note"] = " ".join(notes)
    if counts["failed"] and counts["failed"] < len(fresh):
        op.status = "partial"
        op.error_message_fa = f"{counts['failed']} پست از {len(fresh)} پست جدید استخراج نشد."


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
    shop_categories = await categories.category_names(db, store.id)
    prompt = json.dumps({"seller_note": ref.note, "photo_role": photo_role, "price": ref.price,
                         "shop_categories": shop_categories[:categories.PROMPT_LIMIT]}, ensure_ascii=False)
    async with ctx.step("generate", "ساخت محصول با هوش مصنوعی", request={"images": len(images)}) as s:
        result = await ai.generate_json(task="product_generate", system=ai.PRODUCT_FROM_PHOTO_SYSTEM,
                                        prompt=prompt, images=images, store_id=store.id,
                                        max_tokens=PRODUCT_MAX_TOKENS, image_res="medium")
        s.response = result
    product = Product(store_id=store.id, title=normalize_fa(result.get("title") or "محصول جدید")[:300],
                      description=normalize_fa(result.get("description", "")), price=ref.price or int(result.get("price") or 0),
                      stock_qty=ref.stock_qty or parse_number(str(result.get("quantity") or 0)), brand=result.get("brand", ""), category_hint=result.get("category_hint", ""),
                      category=categories.match(result.get("category"), shop_categories),
                      attributes=result.get("attributes") or {}, tags=(result.get("tags") or [])[:8], source="ai",
                      created_by="ai", status="draft",
                      source_ref={"ai_request_id": ref.id, "field_sources": result.get("field_sources", {})})
    db.add(product)
    await db.flush()
    await product_code.apply_default_code(db, product, ref.note or "")
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
