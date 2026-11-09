"""Invoice maker, the unified orders list and the financial summary.

- Invoice settings live in `store.settings["invoice"]` (who the invoice is from, card details, expiry...).
- An invoice gets a public link (`/i/<code>` on the frontend). The customer opens it with no account, pays card to card
  (or through the seller's own payment link) and uploads the receipt; it then shows in the orders list.
- The orders list merges the assistant's orders (`AiOrder`, source "ai") with the seller's own (`Invoice`, sources
  "invoice" and "chat").
"""

import secrets
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.models import ActivityEvent, AiOrder, ChatAgent, Invoice, Product, Store
from app.routers.chat import _order_out as ai_order_out
from app.services.chat_engine import confirmed_toman
from app.services.media import ALLOWED, save_bytes
from app.services.notify import notify
from app.services.ops import record_event

router = APIRouter(tags=["orders"])

STATUSES = ("awaiting_payment", "payment_submitted", "confirmed", "rejected", "cancelled")
PLATFORMS = ("instagram", "telegram", "bale", "whatsapp", "rubika", "eitaa", "divar", "phone", "in_person", "site", "other")
SOURCE_EVENT = {"invoice": "sale_invoice_toman", "chat": "sale_chat_toman"}

# ---------- settings ----------

DEFAULT_INVOICE_SETTINGS: dict[str, Any] = {
    "seller_name": "", "phone": "", "address": "", "logo_url": "",
    "card_number": "", "sheba": "", "holder": "", "bank_name": "",
    "pay_link": "",  # the seller's own online payment page (Zarinpal link...), optional
    "require_receipt": True,  # the customer must upload a receipt image (else a tracking code is enough)
    "ask_address": True, "ask_phone": True,
    "expire_hours": 72,  # 0 = never
    "default_shipping_toman": 0,
    "note": "",  # printed on every invoice: return policy, delivery time...
    "thank_you": "",  # shown after the customer sends the receipt
    "next_number": 1001,
}


class InvoiceSettingsIn(BaseModel):
    seller_name: str | None = Field(None, max_length=120)
    phone: str | None = Field(None, max_length=30)
    address: str | None = Field(None, max_length=500)
    logo_url: str | None = Field(None, max_length=1000)
    card_number: str | None = Field(None, max_length=40)
    sheba: str | None = Field(None, max_length=40)
    holder: str | None = Field(None, max_length=120)
    bank_name: str | None = Field(None, max_length=60)
    pay_link: str | None = Field(None, max_length=500)
    require_receipt: bool | None = None
    ask_address: bool | None = None
    ask_phone: bool | None = None
    expire_hours: int | None = Field(None, ge=0, le=24 * 60)
    default_shipping_toman: int | None = Field(None, ge=0)
    note: str | None = Field(None, max_length=1500)
    thank_you: str | None = Field(None, max_length=500)
    next_number: int | None = Field(None, ge=1, le=10**9)


async def invoice_settings(db, store: Store) -> dict[str, Any]:
    """The store's invoice settings; empty card details fall back to what the chat assistant was given."""
    out = {**DEFAULT_INVOICE_SETTINGS, **((store.settings or {}).get("invoice") or {})}
    if not out["seller_name"]:
        out["seller_name"] = store.name
    if not out["logo_url"]:
        out["logo_url"] = store.logo_url or ""
    if not (out["card_number"] or out["sheba"]):
        agent = (await db.execute(select(ChatAgent).where(ChatAgent.store_id == store.id))).scalar_one_or_none()
        pay = (agent.payment if agent else None) or {}
        for k in ("card_number", "sheba", "holder"):
            out[k] = out[k] or pay.get(k) or ""
        out["pay_link"] = out["pay_link"] or pay.get("link") or ""
    return out


def _settings_out(s: dict[str, Any]) -> dict[str, Any]:
    return {**s, "ready": bool(s["card_number"] or s["sheba"] or s["pay_link"])}


@router.get("/stores/{store_id}/invoice-settings")
async def get_invoice_settings(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    return _settings_out(await invoice_settings(db, store))


@router.put("/stores/{store_id}/invoice-settings")
async def put_invoice_settings(store_id: int, body: InvoiceSettingsIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    data = body.model_dump(exclude_none=True)
    for k in ("card_number", "sheba"):
        if k in data:
            data[k] = _latin_digits(data[k]).replace(" ", "").replace("-", "").upper()
    if data.get("card_number") and not (data["card_number"].isdigit() and len(data["card_number"]) == 16):
        raise HTTPException(422, "شماره کارت باید ۱۶ رقم باشد.")
    if data.get("sheba"):
        sheba = data["sheba"] if data["sheba"].startswith("IR") else "IR" + data["sheba"]
        if not (len(sheba) == 26 and sheba[2:].isdigit()):
            raise HTTPException(422, "شبا باید IR و ۲۴ رقم باشد.")
        data["sheba"] = sheba
    if data.get("pay_link") and not data["pay_link"].startswith(("http://", "https://")):
        raise HTTPException(422, "لینک پرداخت باید با https:// شروع شود.")
    s = dict(store.settings or {})
    s["invoice"] = {**((s.get("invoice")) or {}), **data}
    store.settings = s
    await db.commit()
    return _settings_out(await invoice_settings(db, store))


# ---------- invoices ----------


def _latin_digits(v: str) -> str:
    return (v or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")).strip()


class LineIn(BaseModel):
    product_id: int | None = None
    title: str = Field("", max_length=300)
    variant: str = Field("", max_length=120)
    qty: int = Field(1, ge=1, le=10000)
    unit_price: int | None = Field(None, ge=0)


class InvoiceIn(BaseModel):
    source: str = "invoice"  # invoice | chat
    platform: str = ""
    customer_name: str = Field("", max_length=160)
    customer_phone: str = Field("", max_length=30)
    customer_address: str = Field("", max_length=1000)
    items: list[LineIn] = Field(min_length=1, max_length=50)
    discount_toman: int = Field(0, ge=0)
    shipping_toman: int | None = Field(None, ge=0)
    note: str = Field("", max_length=1500)
    seller_note: str = Field("", max_length=1000)
    # A chat order the seller already got paid for can be logged as confirmed straight away.
    status: str = "awaiting_payment"


class InvoicePatch(BaseModel):
    status: str | None = None
    seller_note: str | None = Field(None, max_length=1000)
    customer_name: str | None = Field(None, max_length=160)
    customer_phone: str | None = Field(None, max_length=30)
    customer_address: str | None = Field(None, max_length=1000)
    items: list[LineIn] | None = Field(None, min_length=1, max_length=50)
    discount_toman: int | None = Field(None, ge=0)
    shipping_toman: int | None = Field(None, ge=0)
    note: str | None = Field(None, max_length=1500)


def public_link(inv: Invoice) -> str:
    return f"{settings.frontend_base_url.rstrip('/')}/i/{inv.code}"


def _expired(inv: Invoice) -> bool:
    exp = inv.expires_at
    if not exp or inv.status != "awaiting_payment":
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp < utcnow()


def invoice_out(inv: Invoice) -> dict[str, Any]:
    return {"id": inv.id, "number": inv.number, "source": inv.source, "code": inv.code, "platform": inv.platform,
            "link": public_link(inv) if inv.source == "invoice" else "",
            "customer_name": inv.customer_name, "customer_phone": inv.customer_phone,
            "customer_address": inv.customer_address, "items": inv.items or [], "subtotal_toman": inv.subtotal_toman,
            "discount_toman": inv.discount_toman, "shipping_toman": inv.shipping_toman,
            "total_toman": inv.total_toman, "note": inv.note, "seller_note": inv.seller_note, "status": inv.status,
            "expired": _expired(inv), "receipt_url": inv.receipt_url, "payment_ref": inv.payment_ref,
            "payer_note": inv.payer_note, "expires_at": inv.expires_at, "viewed_at": inv.viewed_at,
            "submitted_at": inv.submitted_at, "paid_at": inv.paid_at, "created_at": inv.created_at,
            "updated_at": inv.updated_at}


async def _priced(db, store_id: int, lines: list[LineIn]) -> list[dict[str, Any]]:
    ids = [ln.product_id for ln in lines if ln.product_id]
    products = {p.id: p for p in (await db.execute(
        select(Product).options(selectinload(Product.images))
        .where(Product.id.in_(ids or [0]), Product.store_id == store_id))).scalars()}
    out = []
    for ln in lines:
        p = products.get(ln.product_id or 0)
        if ln.product_id and not p:
            raise HTTPException(404, "محصول پیدا نشد.")
        title = (ln.title or (p.title if p else "")).strip()
        if not title:
            raise HTTPException(422, "نام هر ردیف فاکتور لازم است.")
        price = ln.unit_price if ln.unit_price is not None else (p.price if p else 0)
        out.append({"product_id": p.id if p else None, "title": title, "variant": ln.variant.strip(), "qty": ln.qty,
                    "unit_price": int(price or 0), "image_url": p.images[0].url if p and p.images else ""})
    return out


def _totals(inv: Invoice) -> None:
    inv.subtotal_toman = sum(int(i["unit_price"]) * int(i["qty"]) for i in inv.items or [])
    inv.discount_toman = min(inv.discount_toman or 0, inv.subtotal_toman)
    inv.total_toman = max(0, inv.subtotal_toman - inv.discount_toman + (inv.shipping_toman or 0))


async def _confirm(db, inv: Invoice) -> None:
    """Paid: take the stock (once) and count the sale toward today's income and its source."""
    if not inv.stock_taken:
        for it in inv.items or []:
            p = await db.get(Product, it.get("product_id") or 0) if it.get("product_id") else None
            if p and p.store_id == inv.store_id:
                p.stock_qty = max(0, p.stock_qty - int(it.get("qty") or 0))
        inv.stock_taken = True
    inv.paid_at = utcnow()
    await record_event(db, inv.store_id, "sale_toman", inv.total_toman, payload={"invoice_id": inv.id})
    await record_event(db, inv.store_id, SOURCE_EVENT.get(inv.source, "sale_invoice_toman"), inv.total_toman,
                       payload={"invoice_id": inv.id})


async def _get_invoice(db, user, invoice_id: int) -> Invoice:
    inv = await db.get(Invoice, invoice_id)
    if not inv:
        raise HTTPException(404, "فاکتور پیدا نشد.")
    await get_store_for(db, user, inv.store_id)
    return inv


@router.post("/stores/{store_id}/invoices", status_code=201)
async def create_invoice(store_id: int, body: InvoiceIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    if body.source not in ("invoice", "chat"):
        raise HTTPException(422, "نوع سفارش نامعتبر است.")
    if body.platform and body.platform not in PLATFORMS:
        raise HTTPException(422, "کانال سفارش نامعتبر است.")
    if body.status not in ("awaiting_payment", "confirmed") or (body.source == "invoice" and body.status != "awaiting_payment"):
        raise HTTPException(422, "وضعیت نامعتبر است.")
    conf = await invoice_settings(db, store)
    number = int(conf.get("next_number") or 1001)
    last = await db.scalar(select(func.max(Invoice.number)).where(Invoice.store_id == store_id))
    number = max(number, (last or 0) + 1)
    s = dict(store.settings or {})
    s["invoice"] = {**(s.get("invoice") or {}), "next_number": number + 1}
    store.settings = s
    hours = int(conf.get("expire_hours") or 0)
    inv = Invoice(
        store_id=store_id, number=number, source=body.source, code=secrets.token_urlsafe(9), platform=body.platform,
        customer_name=body.customer_name.strip(), customer_phone=_latin_digits(body.customer_phone),
        customer_address=body.customer_address.strip(), items=await _priced(db, store_id, body.items),
        discount_toman=body.discount_toman,
        shipping_toman=body.shipping_toman if body.shipping_toman is not None else int(conf.get("default_shipping_toman") or 0),
        note=body.note.strip(), seller_note=body.seller_note.strip(), status="awaiting_payment",
        expires_at=utcnow() + timedelta(hours=hours) if hours and body.source == "invoice" else None,
    )
    _totals(inv)
    if inv.total_toman <= 0 and body.source == "invoice":
        raise HTTPException(422, "جمع فاکتور باید بیشتر از صفر باشد.")
    db.add(inv)
    await db.flush()
    if body.source == "chat":
        inv.submitted_at = utcnow()  # a chat order is an order from the start
        if body.status == "confirmed":
            inv.status = "confirmed"
            await _confirm(db, inv)
    await db.commit()
    return invoice_out(inv)


@router.get("/stores/{store_id}/invoices")
async def list_invoices(store_id: int, user: CurrentUser, db: DB, status: str = "", q: str = "", source: str = "invoice",
                        limit: int = 50, offset: int = 0):
    await get_store_for(db, user, store_id)
    stmt = select(Invoice).where(Invoice.store_id == store_id)
    if source:
        stmt = stmt.where(Invoice.source == source)
    if status:
        stmt = stmt.where(Invoice.status == status)
    if q.strip():
        like = f"%{q.strip()}%"
        num = _latin_digits(q)
        stmt = stmt.where(or_(Invoice.customer_name.ilike(like), Invoice.customer_phone.ilike(f"%{num}%"),
                              *([Invoice.number == int(num)] if num.isdigit() else [])))
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = (await db.execute(stmt.order_by(Invoice.id.desc()).limit(min(limit, 100)).offset(offset))).scalars().all()
    counts = dict((await db.execute(select(Invoice.status, func.count()).where(
        Invoice.store_id == store_id, Invoice.source == (source or "invoice")).group_by(Invoice.status))).all())
    return {"total": total, "counts": counts, "items": [invoice_out(i) for i in rows]}


@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: int, user: CurrentUser, db: DB):
    return invoice_out(await _get_invoice(db, user, invoice_id))


@router.patch("/invoices/{invoice_id}")
async def update_invoice(invoice_id: int, body: InvoicePatch, user: CurrentUser, db: DB):
    inv = await _get_invoice(db, user, invoice_id)
    data = body.model_dump(exclude_none=True)
    editable = inv.status in ("awaiting_payment", "rejected")
    for k in ("customer_name", "customer_address", "note", "seller_note"):
        if k in data:
            setattr(inv, k, data[k].strip())
    if "customer_phone" in data:
        inv.customer_phone = _latin_digits(data["customer_phone"])
    if any(k in data for k in ("items", "discount_toman", "shipping_toman")):
        if not editable:
            raise HTTPException(409, "فاکتور پرداخت‌شده را نمی‌شود ویرایش کرد.")
        if body.items is not None:
            inv.items = await _priced(db, inv.store_id, body.items)
        if body.discount_toman is not None:
            inv.discount_toman = body.discount_toman
        if body.shipping_toman is not None:
            inv.shipping_toman = body.shipping_toman
        _totals(inv)
    if body.status and body.status != inv.status:
        if body.status not in STATUSES:
            raise HTTPException(422, "وضعیت نامعتبر است.")
        if inv.status == "confirmed":
            raise HTTPException(409, "سفارش تأییدشده را نمی‌شود برگرداند.")
        if body.status == "confirmed":
            await _confirm(db, inv)
        if body.status == "awaiting_payment" and inv.source == "invoice":
            store = await db.get(Store, inv.store_id)
            hours = int((await invoice_settings(db, store)).get("expire_hours") or 0)
            inv.expires_at = utcnow() + timedelta(hours=hours) if hours else None  # reopened: a fresh deadline
        inv.status = body.status
    await db.commit()
    return invoice_out(inv)


# ---------- public invoice page (no account) ----------


async def _public(db, code: str) -> tuple[Invoice, Store]:
    inv = (await db.execute(select(Invoice).where(Invoice.code == code, Invoice.source == "invoice"))).scalar_one_or_none()
    store = await db.get(Store, inv.store_id) if inv else None
    if not inv or not store or store.archived:
        raise HTTPException(404, "فاکتور پیدا نشد.")
    return inv, store


def _public_out(inv: Invoice, conf: dict[str, Any]) -> dict[str, Any]:
    # Once the receipt is in, or the invoice is paid, cancelled or expired, the link only shows its state.
    closed = inv.status in ("payment_submitted", "confirmed", "cancelled") or _expired(inv)
    pay = {k: "" if closed else conf.get(k) or "" for k in ("card_number", "sheba", "holder", "bank_name", "pay_link")}
    return {
        "number": inv.number, "status": inv.status, "expired": _expired(inv), "closed": closed, "expires_at": inv.expires_at,
        "created_at": inv.created_at, "items": [{k: i.get(k) for k in ("title", "variant", "qty", "unit_price", "image_url")}
                                                for i in inv.items or []],
        "subtotal_toman": inv.subtotal_toman, "discount_toman": inv.discount_toman,
        "shipping_toman": inv.shipping_toman, "total_toman": inv.total_toman, "note": inv.note,
        "customer": {"name": inv.customer_name, "phone": inv.customer_phone, "address": inv.customer_address},
        "seller": {k: conf.get(k) or "" for k in ("seller_name", "phone", "address", "logo_url")},
        "terms": conf.get("note") or "", "thank_you": conf.get("thank_you") or "",
        "payment": pay, "require_receipt": bool(conf.get("require_receipt", True)),
        "ask_address": bool(conf.get("ask_address", True)), "ask_phone": bool(conf.get("ask_phone", True)),
        "receipt_sent": bool(inv.submitted_at), "submitted_at": inv.submitted_at,
    }


@router.get("/public/invoices/{code}")
async def public_invoice(code: str, db: DB):
    inv, store = await _public(db, code)
    if not inv.viewed_at:
        inv.viewed_at = utcnow()
        await db.commit()
    return _public_out(inv, await invoice_settings(db, store))


@router.post("/public/invoices/{code}/receipt")
async def public_receipt(code: str, db: DB, file: UploadFile | None = File(None), payment_ref: str = Form(""),
                         name: str = Form(""), phone: str = Form(""), address: str = Form(""), note: str = Form("")):
    inv, store = await _public(db, code)
    conf = await invoice_settings(db, store)
    if inv.status in ("confirmed", "cancelled", "payment_submitted"):
        raise HTTPException(409, {"cancelled": "این فاکتور لغو شده است.", "confirmed": "پرداخت این فاکتور تأیید شده است.",
                                  "payment_submitted": "فیش این فاکتور قبلاً ارسال شده و منتظر تأیید فروشنده است."}[inv.status])
    if _expired(inv):
        raise HTTPException(409, "مهلت پرداخت این فاکتور تمام شده؛ از فروشنده فاکتور تازه بخواه.")
    ref = _latin_digits(payment_ref)[:120]
    url = ""
    if file is not None and file.filename:
        if file.content_type not in ALLOWED:
            raise HTTPException(422, "فیش باید عکس JPG، PNG یا WEBP باشد.")
        data = await file.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(422, "حجم عکس فیش حداکثر ۸ مگابایت است.")
        url = save_bytes(inv.store_id, data, file.filename or "", file.content_type or "")
    if conf.get("require_receipt", True) and not url and not inv.receipt_url:
        raise HTTPException(422, "عکس فیش واریز را بارگذاری کن.")
    if not url and not ref and not inv.receipt_url:
        raise HTTPException(422, "عکس فیش یا کد پیگیری واریز را وارد کن.")
    name, phone, address = name.strip()[:160], _latin_digits(phone)[:30], address.strip()[:1000]
    if conf.get("ask_phone", True) and not (phone or inv.customer_phone):
        raise HTTPException(422, "شماره موبایل را وارد کن.")
    if conf.get("ask_address", True) and not (address or inv.customer_address):
        raise HTTPException(422, "آدرس تحویل را وارد کن.")
    inv.customer_name = name or inv.customer_name
    inv.customer_phone = phone or inv.customer_phone
    inv.customer_address = address or inv.customer_address
    inv.receipt_url = url or inv.receipt_url
    inv.payment_ref = ref or inv.payment_ref
    inv.payer_note = note.strip()[:1000]
    inv.status = "payment_submitted"
    inv.submitted_at = utcnow()
    await notify(db, inv.store_id, "approval", f"فیش فاکتور {inv.number} رسید",
                 f"{inv.customer_name or 'مشتری'} · {inv.total_toman:,} تومان. بررسی و تأیید کن.",
                 link=f"/app/s/{inv.store_id}/orders?o=inv-{inv.id}", image_url=inv.receipt_url)
    await db.commit()
    return _public_out(inv, conf)


# ---------- the unified orders list ----------


def _ai_item(o: AiOrder) -> dict[str, Any]:
    d = ai_order_out(o)
    return {**d, "key": f"ai-{o.id}", "kind": "ai", "number": o.id, "ref": o.tracking_code or "",
            "subtotal_toman": o.total_toman, "discount_toman": 0, "shipping_toman": 0, "link": "",
            "submitted_at": o.created_at, "payer_note": o.receipt_text or ""}


def _inv_item(i: Invoice) -> dict[str, Any]:
    d = invoice_out(i)
    return {**d, "key": f"inv-{i.id}", "kind": i.source, "ref": "", "conversation_id": None, "customer_username": "",
            "extras": {}, "receipt_text": i.payer_note, "is_test": False, "tracking_code": "",
            "due_toman": i.total_toman if i.status in ("awaiting_payment", "payment_submitted") else 0}


def _order_at(i: Invoice):
    return i.submitted_at or i.created_at


def _aware(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@router.get("/stores/{store_id}/orders")
async def unified_orders(store_id: int, user: CurrentUser, db: DB, source: str = "", status: str = "", q: str = "",
                         day: date | None = None, limit: int = 30, offset: int = 0):
    """Every order of the store, newest first: the assistant's (ai), the seller's chat orders (chat) and invoices the
    customer has paid or sent a receipt for (invoice)."""
    await get_store_for(db, user, store_id)
    limit = min(max(limit, 1), 100)
    want = limit + offset
    lo = datetime.combine(day, time.min, tzinfo=timezone.utc) if day else None
    q = q.strip()
    num = _latin_digits(q)
    items: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    if source in ("", "ai"):
        st = select(AiOrder).where(AiOrder.store_id == store_id, ~AiOrder.is_test)
        if status:
            st = st.where(AiOrder.status == status)
        if lo:
            st = st.where(AiOrder.created_at >= lo, AiOrder.created_at < lo + timedelta(days=1))
        if q:
            st = st.where(or_(AiOrder.customer_name.ilike(f"%{q}%"), AiOrder.customer_phone.ilike(f"%{num}%"),
                              AiOrder.tracking_code.ilike(f"%{q}%"), *([AiOrder.id == int(num)] if num.isdigit() else [])))
        totals["ai"] = int(await db.scalar(select(func.count()).select_from(st.subquery())) or 0)
        items += [_ai_item(o) for o in (await db.execute(st.order_by(AiOrder.created_at.desc()).limit(want))).scalars()]
    if source in ("", "invoice", "chat"):
        st = select(Invoice).where(Invoice.store_id == store_id, Invoice.submitted_at.is_not(None))
        if source:
            st = st.where(Invoice.source == source)
        if status:
            st = st.where(Invoice.status == status)
        if lo:
            st = st.where(Invoice.submitted_at >= lo, Invoice.submitted_at < lo + timedelta(days=1))
        if q:
            st = st.where(or_(Invoice.customer_name.ilike(f"%{q}%"), Invoice.customer_phone.ilike(f"%{num}%"),
                              *([Invoice.number == int(num)] if num.isdigit() else [])))
        sub = st.subquery()
        for src, n in (await db.execute(select(sub.c.source, func.count()).group_by(sub.c.source))).all():
            totals[src] = int(n)
        items += [_inv_item(i) for i in (await db.execute(st.order_by(Invoice.submitted_at.desc()).limit(want))).scalars()]
    items.sort(key=lambda d: _aware(d["submitted_at"] or d["created_at"]), reverse=True)
    # Counts per status over every source, for the filter tabs.
    counts: dict[str, int] = {}
    for s, n in (await db.execute(select(AiOrder.status, func.count()).where(
            AiOrder.store_id == store_id, ~AiOrder.is_test).group_by(AiOrder.status))).all():
        counts[s] = counts.get(s, 0) + int(n)
    for s, n in (await db.execute(select(Invoice.status, func.count()).where(
            Invoice.store_id == store_id, Invoice.submitted_at.is_not(None)).group_by(Invoice.status))).all():
        counts[s] = counts.get(s, 0) + int(n)
    by_source = {"ai": int(await db.scalar(select(func.count()).select_from(AiOrder).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test)) or 0)}
    for s, n in (await db.execute(select(Invoice.source, func.count()).where(
            Invoice.store_id == store_id, Invoice.submitted_at.is_not(None)).group_by(Invoice.source))).all():
        by_source[s] = int(n)
    return {"total": sum(totals.values()), "counts": counts, "sources": by_source,
            "items": items[offset:offset + limit]}


# ---------- money: income by source, per day ----------

SOURCES = ("ai", "chat", "invoice")


async def finance_series(db, store_id: int, start: date, end: date) -> dict[str, dict[str, int]]:
    """Per day: income (confirmed toman) and orders, by source."""
    blank = {f"{s}_{k}": 0 for s in SOURCES for k in ("toman", "orders")}
    days: dict[str, dict[str, int]] = {}
    for d, typ, n in (await db.execute(
            select(ActivityEvent.day, ActivityEvent.type, func.sum(ActivityEvent.count))
            .where(ActivityEvent.store_id == store_id, ActivityEvent.day >= start, ActivityEvent.day <= end,
                   ActivityEvent.type.in_(["sale_toman", *SOURCE_EVENT.values()]))
            .group_by(ActivityEvent.day, ActivityEvent.type))).all():
        row = days.setdefault(d.isoformat(), dict(blank))
        row[{"sale_toman": "all", "sale_invoice_toman": "invoice", "sale_chat_toman": "chat"}[typ] + "_toman"] = int(n)
    for row in days.values():  # the assistant's share is the rest of the day's income
        row["ai_toman"] = max(0, row.pop("all_toman", 0) - row["invoice_toman"] - row["chat_toman"])
    lo = datetime.combine(start, time.min, tzinfo=timezone.utc)
    hi = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
    oday = func.date(AiOrder.created_at)
    for d, n in (await db.execute(select(oday, func.count()).where(
            AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.created_at >= lo, AiOrder.created_at < hi,
            AiOrder.status.notin_(["cancelled", "rejected"])).group_by(oday))).all():
        days.setdefault(str(d)[:10], dict(blank))["ai_orders"] = int(n)
    iday = func.date(Invoice.submitted_at)
    for d, src, n in (await db.execute(select(iday, Invoice.source, func.count()).where(
            Invoice.store_id == store_id, Invoice.submitted_at >= lo, Invoice.submitted_at < hi,
            Invoice.status.notin_(["cancelled", "rejected"])).group_by(iday, Invoice.source))).all():
        days.setdefault(str(d)[:10], dict(blank))[f"{src}_orders"] = int(n)
    for row in days.values():
        for k, v in blank.items():
            row.setdefault(k, v)
    return days


@router.get("/stores/{store_id}/finance")
async def finance(store_id: int, user: CurrentUser, db: DB, start: date | None = None, end: date | None = None):
    await get_store_for(db, user, store_id)
    end = end or utcnow().date()
    start = start or end - timedelta(days=29)
    if (end - start).days > 400:
        raise HTTPException(422, "بازه حداکثر یک سال است.")
    days = await finance_series(db, store_id, start, end)
    span = (end - start).days + 1
    prev = await finance_series(db, store_id, start - timedelta(days=span), start - timedelta(days=1))

    def sums(series):
        out = {s: {"toman": sum(v[f"{s}_toman"] for v in series.values()),
                   "orders": sum(v[f"{s}_orders"] for v in series.values())} for s in SOURCES}
        return out

    by = sums(days)
    income = sum(v["toman"] for v in by.values())
    orders = sum(v["orders"] for v in by.values())
    sale_days = sum(1 for v in days.values() if any(v[f"{s}_toman"] or v[f"{s}_orders"] for s in SOURCES))
    # Money on the way: receipts the seller hasn't checked yet.
    pending = int(await db.scalar(select(func.coalesce(func.sum(Invoice.total_toman), 0)).where(
        Invoice.store_id == store_id, Invoice.status == "payment_submitted")) or 0)
    for o in (await db.execute(select(AiOrder).where(AiOrder.store_id == store_id, ~AiOrder.is_test,
                                                     AiOrder.status == "payment_submitted"))).scalars():
        pending += max(0, int(o.total_toman or 0) - confirmed_toman(o))
    open_invoices = int(await db.scalar(select(func.coalesce(func.sum(Invoice.total_toman), 0)).where(
        Invoice.store_id == store_id, Invoice.source == "invoice", Invoice.status == "awaiting_payment",
        or_(Invoice.expires_at.is_(None), Invoice.expires_at > utcnow()))) or 0)
    prev_income = sum(v["toman"] for v in sums(prev).values())
    return {"start": start, "end": end, "days": days, "by_source": by,
            "totals": {"income": income, "orders": orders, "avg_order": round(income / orders) if orders else 0,
                       "sale_days": sale_days, "no_sale_days": span - sale_days, "pending_toman": pending,
                       "open_invoices_toman": open_invoices, "prev_income": prev_income}}
