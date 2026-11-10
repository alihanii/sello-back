"""The store's own chat page («لینک چت»): one public link per store for the Instagram / Telegram bio.

- Seller side (`/stores/{id}/chat-link`): the link (made on first look as `shop<id>`), its address, on/off, greeting.
- Public side (`/public/chat/{slug}`, no account): the store's name, logo and greeting, then the customer's chat. The
  customer gives a name first (the page can prefill it from `?name=`). Messages are saved in the store's chats with
  platform "link" and answered by the same assistant as Telegram, Bale and the site widget; the seller can answer
  from the panel and the page picks that up by polling. Orders are AiOrders, so they show in the orders list.
"""

import asyncio
import json
import re
import secrets
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import timedelta

from sqlalchemy import func, select

from app.core.db import SessionLocal, utcnow

from app.core.config import settings
from app.deps import DB, CurrentUser, get_store_for
from app.models import ChatConversation, ChatLink, ChatMessage, Store
from app.routers.site import RESERVED, _limit, abs_media
from app.services import chat_runtime as rt
from app.services import live
from app.services.chat_engine import get_agent
from app.services.chat_menu import MAIN_BUTTONS, SEND_PHONE
from app.services.media import ALLOWED, save_bytes

router = APIRouter(tags=["chat-link"])

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,30}[a-z0-9])$")
VISITOR_RE = re.compile(r"[\w-]{16,40}")
DEFAULT_GREETING = "سلام! خوش اومدی 🌱 هر سؤالی دربارهٔ محصولات داری بپرس؛ سفارشت رو هم همین‌جا ثبت می‌کنیم."
PLATFORM = "link"
# The link is public, so one chat can't spend the store's AI: at most this many assistant replies an hour; after that
# the messages still reach the seller in the panel and the page says so.
AI_REPLIES_PER_HOUR = 30
CAP_NOTICE = "پیامت رسید ✅ همکارمان به‌زودی خودش جواب می‌دهد."


def link_url(link: ChatLink) -> str:
    return f"{settings.frontend_base_url.rstrip('/')}/c/{link.slug}"


async def _link(db, store: Store) -> ChatLink:
    link = (await db.execute(select(ChatLink).where(ChatLink.store_id == store.id))).scalar_one_or_none()
    if link:
        return link
    slug = f"shop{store.id}"
    if (await db.execute(select(ChatLink.id).where(ChatLink.slug == slug))).first():
        slug = f"shop{store.id}-{secrets.token_hex(2)}"
    link = ChatLink(store_id=store.id, slug=slug, enabled=True, greeting="")
    db.add(link)
    await db.commit()
    await db.refresh(link)
    return link


async def _seller_out(db, store: Store, link: ChatLink) -> dict[str, Any]:
    agent = await get_agent(db, store.id)
    return {"slug": link.slug, "enabled": link.enabled, "greeting": link.greeting, "default_greeting": DEFAULT_GREETING,
            "url": link_url(link), "assistant_on": bool(agent.enabled)}


class LinkIn(BaseModel):
    slug: str | None = Field(None, max_length=40)
    enabled: bool | None = None
    greeting: str | None = Field(None, max_length=300)


@router.get("/stores/{store_id}/chat-link")
async def get_link(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    return await _seller_out(db, store, await _link(db, store))


@router.put("/stores/{store_id}/chat-link")
async def put_link(store_id: int, body: LinkIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    link = await _link(db, store)
    if body.slug is not None:
        slug = body.slug.strip().lower()
        if not SLUG_RE.fullmatch(slug) or slug in RESERVED:
            raise HTTPException(422, "آدرس لینک فقط حروف انگلیسی کوچک، عدد و خط تیره (۳ تا ۳۲ حرف) باشد.")
        taken = (await db.execute(select(ChatLink.id).where(ChatLink.slug == slug, ChatLink.id != link.id))).first()
        if taken:
            raise HTTPException(409, "این آدرس را فروشگاه دیگری برداشته؛ یکی دیگر انتخاب کن.")
        link.slug = slug
    if body.enabled is not None:
        link.enabled = body.enabled
    if body.greeting is not None:
        link.greeting = body.greeting.strip()[:300]
    await db.commit()
    return await _seller_out(db, store, link)


# ---------- public ----------


async def _public(db, slug: str) -> tuple[ChatLink, Store]:
    link = (await db.execute(select(ChatLink).where(ChatLink.slug == slug.lower()))).scalar_one_or_none()
    store = await db.get(Store, link.store_id) if link else None
    if not link or not store or store.archived:
        raise HTTPException(404, "این صفحهٔ چت پیدا نشد.")
    if not link.enabled:
        raise HTTPException(403, "چت این فروشگاه فعلاً خاموش است.")
    return link, store


def _msg(m: ChatMessage) -> dict[str, Any]:
    return {"id": m.id, "role": m.role, "text": m.text, "image_url": abs_media(m.image_url),
            "buttons": m.buttons or [], "created_at": m.created_at}


async def _conv(db, store: Store, visitor: str) -> ChatConversation | None:
    if not VISITOR_RE.fullmatch(visitor or ""):
        return None
    return (await db.execute(select(ChatConversation).where(
        ChatConversation.store_id == store.id, ChatConversation.platform == PLATFORM,
        ChatConversation.external_chat_id == visitor))).scalar_one_or_none()


@router.get("/public/chat/{slug}")
async def public_link(slug: str, db: DB):
    link, store = await _public(db, slug)
    agent = await get_agent(db, store.id)
    return {"slug": link.slug, "name": store.name, "logo_url": abs_media(store.logo_url or ""),
            "greeting": link.greeting or DEFAULT_GREETING, "assistant_on": bool(agent.enabled),
            # Like the Telegram / Bale bot: the cart and order buttons stay under the chat while orders are on, and a
            # reply's «📱 ارسال شماره‌ی من» opens a phone box on the page instead of sharing the messenger contact.
            "menu": MAIN_BUTTONS if agent.orders_enabled else [], "phone_button": SEND_PHONE}


@router.get("/public/chat/{slug}/messages")
async def history(slug: str, db: DB, visitor: str = Query("", max_length=40), after: int = 0):
    """The customer's chat (polled while the page is open, for the seller's replies)."""
    _, store = await _public(db, slug)
    conv = await _conv(db, store, visitor)
    if not conv:
        return {"visitor": visitor, "name": "", "messages": []}
    rows = (await db.execute(select(ChatMessage).where(
        ChatMessage.conversation_id == conv.id, ChatMessage.id > after,
        ChatMessage.role.in_(["customer", "ai", "seller"])).order_by(ChatMessage.id).limit(200))).scalars().all()
    return {"visitor": visitor, "name": conv.customer_name, "messages": [_msg(m) for m in rows]}


STREAM_IDLE = 60 * 60  # an open page with nothing new for an hour is let go; it opens the stream again when used
STREAM_PING = 25


async def _after(db, conv_id: int, after: int) -> list[ChatMessage]:
    return list((await db.execute(select(ChatMessage).where(
        ChatMessage.conversation_id == conv_id, ChatMessage.id > after,
        ChatMessage.role.in_(["customer", "ai", "seller"])).order_by(ChatMessage.id).limit(200))).scalars().all())


@router.get("/public/chat/{slug}/stream")
async def stream(slug: str, request: Request, db: DB, visitor: str = Query("", max_length=40), after: int = 0):
    """Server-Sent Events: the chat's new messages (the assistant's and the seller's) as they are saved, so the page
    never asks again and again. Each event is a JSON list of messages; "bye" after an hour with nothing new."""
    _, store = await _public(db, slug)
    conv = await _conv(db, store, visitor)
    if not conv:
        raise HTTPException(404, "این گفتگو پیدا نشد.")
    conv_id = conv.id
    await db.close()

    async def events():
        last, idle = after, 0.0
        with live.watch(conv_id) as woke:
            while not await request.is_disconnected():
                woke.clear()  # before reading, so a message saved meanwhile wakes the next round
                async with SessionLocal() as s:
                    rows = await _after(s, conv_id, last)
                if rows:
                    last, idle = rows[-1].id, 0.0
                    yield f"data: {json.dumps([_msg(m) for m in rows], ensure_ascii=False, default=str)}\n\n"
                try:
                    await asyncio.wait_for(woke.wait(), STREAM_PING)
                except TimeoutError:
                    idle += STREAM_PING
                    if idle >= STREAM_IDLE:
                        yield "event: bye\ndata: {}\n\n"
                        return
                    yield ": ping\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SendIn(BaseModel):
    visitor: str = Field("", max_length=40)
    name: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1000)


async def _turn(db, request: Request, slug: str, visitor: str, name: str, text: str, image_url_of=None):
    _limit(request, "chat-link", 40, 300)
    _, store = await _public(db, slug)
    name = " ".join((name or "").split())[:80]
    if not name:
        raise HTTPException(422, "اسمت را بنویس.")
    image_url = await image_url_of(store) if image_url_of else ""
    conv = await _conv(db, store, visitor)
    visitor = visitor if conv else secrets.token_urlsafe(16)
    if not conv:
        conv = ChatConversation(store_id=store.id, bot_id=None, platform=PLATFORM, mode="web",
                                external_chat_id=visitor, customer_name=name, state={})
        db.add(conv)
        await db.flush()
    elif name != conv.customer_name:
        conv.customer_name = name
    customer = rt.add_message(db, conv, role="customer", text=(text or "").strip()[:1000], image_url=image_url)
    conv.seller_unread = True
    await db.commit()
    recent = await db.scalar(select(func.count()).select_from(ChatMessage).where(
        ChatMessage.conversation_id == conv.id, ChatMessage.role == "ai",
        ChatMessage.created_at >= utcnow() - timedelta(hours=1)))
    if recent >= AI_REPLIES_PER_HOUR:
        return {"visitor": visitor, "messages": [_msg(customer)], "notice": CAP_NOTICE}
    replies = await rt.answer(conv.id, send=False)
    return {"visitor": visitor, "messages": [_msg(customer)] + [_msg(m) for m in replies], "notice": ""}


@router.post("/public/chat/{slug}/messages")
async def send(slug: str, body: SendIn, request: Request, db: DB):
    """A customer's message (typed, or a button tapped): saved in the store's chats and answered by the store's
    assistant, or later by the seller from the panel."""
    return await _turn(db, request, slug, body.visitor, body.name, body.text)


@router.post("/public/chat/{slug}/photo")
async def send_photo(slug: str, request: Request, db: DB, file: UploadFile = File(...), name: str = Form(...),
                     visitor: str = Form(""), text: str = Form("")):
    """A photo from the customer, as on Telegram: a payment receipt, a design to print, or a product to find."""
    if file.content_type not in ALLOWED:
        raise HTTPException(422, "فقط عکس JPG، PNG، WEBP یا GIF مجاز است.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(422, "حجم عکس حداکثر ۱۰ مگابایت است.")

    async def upload(store: Store) -> str:
        return save_bytes(store.id, data, file.filename or "", file.content_type or "")

    return await _turn(db, request, slug, visitor[:40], name[:80], text[:1000], upload)
