"""Rule engine: "when <trigger>, if <conditions>, then publish to <channels> (auto or approval)"."""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db import utcnow
from app.models import Approval, Automation, Channel, ChannelListing, Product
from app.services.ops import enqueue


def matches(product: Product, cond: dict) -> bool:
    if cond.get("sources") and product.source not in cond["sources"]:
        return False
    if cond.get("has_price") and product.price <= 0:
        return False
    if cond.get("min_images") and len(product.images) < int(cond["min_images"]):
        return False
    if cond.get("in_stock") and product.stock_qty <= 0:
        return False
    if cond.get("tags") and not set(cond["tags"]) & set(product.tags or []):
        return False
    return product.status == "ready"


async def queue_publish(db: AsyncSession, product: Product, channel: Channel, *, trigger: str, actor: str) -> int | None:
    listing = (await db.execute(select(ChannelListing).where(
        ChannelListing.product_id == product.id, ChannelListing.channel_id == channel.id))).scalar_one_or_none()
    if not listing:
        listing = ChannelListing(product_id=product.id, channel_id=channel.id)
        db.add(listing)
        await db.flush()
    if listing.status in ("queued", "publishing"):
        return None  # never run two publishes of the same product × channel at once
    listing.status = "queued"
    op = await enqueue(db, store_id=product.store_id, type="publish", channel_id=channel.id, product_id=product.id,
                       ref_type="listing", ref_id=listing.id, trigger=trigger, actor=actor,
                       title_fa=f"انتشار «{product.title[:60]}» در {channel.display_name or channel.platform}")
    return op.id


async def fire(db: AsyncSession, store_id: int, trigger: str, product_id: int) -> None:
    rules = (await db.execute(select(Automation).where(
        Automation.store_id == store_id, Automation.trigger == trigger, Automation.is_active))).scalars().all()
    if not rules:
        return
    product = (await db.execute(select(Product).where(Product.id == product_id)
                                .options(selectinload(Product.images)))).scalar_one()
    for rule in rules:
        if not matches(product, rule.conditions or {}):
            continue
        await apply_rule(db, rule, product)


async def apply_rule(db: AsyncSession, rule: Automation, product: Product) -> None:
    channel_ids = (rule.actions or {}).get("channel_ids", [])
    if (rule.actions or {}).get("mode", "approval") == "approval":
        db.add(Approval(store_id=rule.store_id, product_id=product.id, automation_id=rule.id, channel_ids=channel_ids))
    else:
        for cid in channel_ids:
            ch = await db.get(Channel, cid)
            if ch and ch.status == "connected":
                await queue_publish(db, product, ch, trigger="automation", actor="ai")
    rule.last_run_at = utcnow()
    rule.run_count += 1


async def run_scheduled(db: AsyncSession) -> None:
    """`schedule` rules: once a day at `schedule_hour` (Tehran), publish the oldest unpublished match."""
    now = utcnow()
    tehran_hour = (now + timedelta(hours=3, minutes=30)).hour
    rules = (await db.execute(select(Automation).where(Automation.trigger == "schedule", Automation.is_active))).scalars().all()
    for rule in rules:
        if int((rule.actions or {}).get("schedule_hour", 10)) != tehran_hour:
            continue
        if rule.last_run_at and now - rule.last_run_at < timedelta(hours=20):
            continue
        channel_ids = set((rule.actions or {}).get("channel_ids", []))
        products = (await db.execute(select(Product).where(Product.store_id == rule.store_id, Product.status == "ready")
                                     .options(selectinload(Product.images), selectinload(Product.listings))
                                     .order_by(Product.id))).scalars().all()
        for p in products:
            published = {l.channel_id for l in p.listings if l.status in ("published", "queued", "publishing")}
            if channel_ids - published and matches(p, rule.conditions or {}):
                await apply_rule(db, rule, p)
                break
