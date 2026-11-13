"""Digikala after publishing: review state of each listing, price + stock once Digikala approves, and the seller's
orders into the unified orders list.

Digikala sends sellers no webhooks, so every connected Digikala channel is read on a timer (admin › integrations ›
دیجی‌کالا: on/off and the interval). A seller can also press «همگام‌سازی» on the channel. Nothing here shows in the
activity feed unless something happened (a product approved, a new order), so the timer doesn't flood it.
"""

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.db import SessionLocal, utcnow
from app.models import Channel, ChannelListing, Invoice, Product
from app.platforms.base import PlatformError
from app.platforms.digikala import REVIEW_FA, digikala
from app.services.integrations import digikala_app
from app.services.notify import notify

log = logging.getLogger(__name__)


class _QuietCtx:
    """The adapter's step logger, without an operation behind it: steps are kept in memory for the sync's result."""

    def __init__(self, db):
        self.db = db
        self.steps: list[dict[str, Any]] = []

    @asynccontextmanager
    async def step(self, key, title, request=None):
        s = SimpleNamespace(response=None, message_fa="", request=request)
        try:
            yield s
            self.steps.append({"key": key, "ok": True})
        except Exception as exc:
            detail = getattr(exc, "detail", None)
            self.steps.append({"key": key, "ok": False, "error": getattr(exc, "message_fa", str(exc))[:300],
                               "request": request, "detail": str(detail)[:1500] if detail else ""})
            raise


def _dkp(remote_id: str) -> int | None:
    v = (remote_id or "").upper().removeprefix("DKP-")
    return int(v) if v.isdigit() else None


def _when(v: str) -> datetime:
    if v:
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00").replace(" ", "T"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return utcnow()


async def sync_listings(db, ctx, channel: Channel) -> dict[str, int]:
    rows = (await db.execute(select(ChannelListing).where(
        ChannelListing.channel_id == channel.id, ChannelListing.status == "published"))).scalars().all()
    todo = {}
    for ln in rows:
        pid = _dkp(ln.remote_id)
        if pid and (ln.remote_status or "") != "live":
            todo[pid] = ln
    if not todo:
        return {"checked": 0, "approved": 0, "live": 0}
    states = await digikala.review_statuses(ctx, channel, set(todo))
    approved = live = 0
    for pid, ln in todo.items():
        st = states.get(pid)
        if not st:
            continue
        before = ln.remote_status or ""
        ln.remote_status, ln.remote_status_fa, ln.last_synced_at = st["state"], st["label"][:120], utcnow()
        meta = dict(ln.remote_meta or {})
        if st["state"] == "approved" and not meta.get("variant_id"):
            product = await db.get(Product, ln.product_id)
            try:
                res = await digikala.create_variant(ctx, channel, pid, product)
                meta.update({**res, "variant_error": ""})
                ln.remote_status, ln.remote_status_fa = "live", REVIEW_FA["live"]
                live += 1
                await notify(db, channel.store_id, "listing_live", f"«{product.title[:60]}» در دیجی‌کالا فعال شد",
                             f"قیمت {product.price:,} تومان و موجودی {product.stock_qty} ثبت شد.",
                             link=f"/app/s/{channel.store_id}/products/{product.id}")
            except PlatformError as exc:
                if exc.code == "expired":
                    raise
                if meta.get("variant_error") != exc.message_fa:
                    await notify(db, channel.store_id, "listing_error", f"ثبت قیمت «{product.title[:60]}» در دیجی‌کالا انجام نشد",
                                 exc.message_fa, link=f"/app/s/{channel.store_id}/products/{product.id}")
                meta["variant_error"] = exc.message_fa
                ln.last_error_fa = exc.message_fa
        elif st["state"] != before and st["state"] in ("approved", "needs_edit", "rejected"):
            approved += st["state"] == "approved"
            product = await db.get(Product, ln.product_id)
            await notify(db, channel.store_id, "listing_status", f"دیجی‌کالا: «{product.title[:60]}» {st['label']}",
                         "", link=f"/app/s/{channel.store_id}/products/{product.id}")
        ln.remote_meta = meta
    return {"checked": len(todo), "approved": approved, "live": live}


async def sync_orders(db, ctx, channel: Channel) -> dict[str, int]:
    from app.routers.orders import _confirm  # the same sale bookkeeping as the seller's own orders

    orders = await digikala.fetch_orders(ctx, channel)
    if not orders:
        return {"orders": 0, "new": 0, "updated": 0}
    ids = [o["id"] for o in orders]
    have = {i.external_id: i for i in (await db.execute(select(Invoice).where(
        Invoice.store_id == channel.store_id, Invoice.source == "digikala", Invoice.external_id.in_(ids)))).scalars()}
    # Our products by their Digikala product id, so stock and pictures follow the order.
    listings = (await db.execute(select(ChannelListing).where(ChannelListing.channel_id == channel.id)
                                 .options(selectinload(ChannelListing.product).selectinload(Product.images)))).scalars().all()
    by_dkp = {_dkp(ln.remote_id): ln.product for ln in listings if _dkp(ln.remote_id)}
    first_sync = not (channel.config or {}).get("orders_synced_at")
    new = updated = 0
    for o in orders:
        inv = have.get(o["id"])
        if inv:
            if (inv.external_status or "") != o["status_code"] or (inv.external_status_fa or "") != o["status_fa"]:
                inv.external_status, inv.external_status_fa = o["status_code"], o["status_fa"]
                if o["status"] == "cancelled" and inv.status != "cancelled":
                    inv.status = "cancelled"
                updated += 1
            continue
        items = []
        for it in o["items"]:
            p = by_dkp.get(it["remote_product_id"])
            items.append({"product_id": p.id if p else None, "title": p.title if p else it["title"],
                          "variant": f"DKPC-{it['variant_id']}" if it.get("variant_id") else "", "qty": it["qty"],
                          "unit_price": it["unit_price"] or (p.price if p else 0),
                          "image_url": (p.images[0].url if p and p.images else it["image_url"])})
        at = _when(o["created_at"])
        inv = Invoice(store_id=channel.store_id, number=0, source="digikala", code=secrets.token_urlsafe(9),
                      platform="digikala", customer_name=o["customer_name"][:160], customer_address=o["city"],
                      items=items, status=o["status"], external_id=o["id"], channel_id=channel.id,
                      external_status=o["status_code"], external_status_fa=o["status_fa"],
                      payment_ref=o["code"][:120], submitted_at=at, seller_note="")
        inv.subtotal_toman = inv.total_toman = sum(int(i["unit_price"]) * int(i["qty"]) for i in items)
        db.add(inv)
        await db.flush()
        if inv.status == "confirmed" and not first_sync:
            await _confirm(db, inv)  # takes the stock and counts the sale
        else:
            inv.stock_taken = True  # orders from before the channel was synced: history only
        new += 1
        if not first_sync:
            await notify(db, channel.store_id, "order_new", f"سفارش تازه از دیجی‌کالا ({o['code']})",
                         "، ".join(f"{i['title'][:40]} ×{i['qty']}" for i in items)[:300],
                         link=f"/app/s/{channel.store_id}/orders")
    return {"orders": len(orders), "new": new, "updated": updated}


async def sync_channel(db, channel: Channel) -> dict[str, Any]:
    """Listings then orders for one Digikala channel. Raises PlatformError("expired") when the session ended."""
    ctx = _QuietCtx(db)
    out: dict[str, Any] = {}
    errors = []
    try:
        out["listings"] = await sync_listings(db, ctx, channel)
    except PlatformError as exc:
        if exc.code == "expired":
            raise
        errors.append(exc.message_fa)
    try:
        out["orders"] = await sync_orders(db, ctx, channel)
        channel.config = {**(channel.config or {}), "orders_synced_at": utcnow().isoformat()}
    except PlatformError as exc:
        if exc.code == "expired":
            raise
        errors.append(exc.message_fa)
    # The failed steps with Digikala's own answer, so a changed panel endpoint can be fixed from the log.
    failed = [st for st in ctx.steps if not st["ok"]][-6:]
    channel.config = {**(channel.config or {}), "synced_at": utcnow().isoformat(), "sync_error": "؛ ".join(errors)[:500],
                      "sync_result": {k: v for k, v in out.items() if k != "errors"}, "sync_failed_steps": failed}
    out["errors"] = errors
    return out


async def sync_one(channel_id: int) -> dict[str, Any]:
    async with SessionLocal() as db:
        ch = await db.get(Channel, channel_id)
        if not ch or ch.platform != "digikala" or ch.status != "connected":
            return {}
        try:
            res = await sync_channel(db, ch)
        except PlatformError as exc:
            await db.rollback()
            ch = await db.get(Channel, channel_id)
            ch.status, ch.last_error = "expired", exc.message_fa
            await notify(db, ch.store_id, "channel_error", f"مشکل در کانال «{ch.display_name or 'دیجی‌کالا'}»",
                         exc.message_fa, link=f"/app/s/{ch.store_id}/channels")
            res = {"errors": [exc.message_fa]}
        await db.commit()
        return res


def _due(ch: Channel, minutes: int, now: datetime) -> bool:
    last = (ch.config or {}).get("synced_at")
    return not last or _when(last) + timedelta(minutes=minutes) <= now


async def poller_loop(stop: asyncio.Event) -> None:
    from app.services.chat_runtime import _sleep

    while not stop.is_set():
        try:
            async with SessionLocal() as db:
                app = await digikala_app(db)
                chans = [] if not app["enabled"] else (await db.execute(select(Channel).where(
                    Channel.platform == "digikala", Channel.status == "connected"))).scalars().all()
                now = utcnow()
                due = [c.id for c in chans if _due(c, app["sync_minutes"], now)]
            for cid in due:
                if stop.is_set():
                    break
                try:
                    await sync_one(cid)
                except Exception:  # noqa: BLE001
                    log.exception("digikala sync failed for channel %s", cid)
        except Exception:  # noqa: BLE001
            log.exception("digikala poller loop error")
        await _sleep(stop, 60)
