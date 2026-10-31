"""Sales chat assistant API (spec §11): settings, connections, test chat, conversations, AI orders, usage."""

import secrets
from datetime import datetime, time, timedelta, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select

from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.deps import DB, AdminUser, CurrentUser, get_store_for
from app.models import (
    AiOrder, Channel, ChatAgent, ChatBot, ChatConversation, ChatMessage, ChatReference, Product, Store, User,
)
from app.platforms.base import PlatformError
from app.services import chat_messenger as tg
from app.services import chat_runtime as rt
from app.services import chat_faq
from app.services.ai import AIError, route_problem
from app.services.chat_engine import (
    DEFAULT_AWAY, DEFAULT_OFF_TOPIC, DEFAULT_WELCOME, chat_limit, get_agent, month_ai_messages,
)

router = APIRouter(tags=["chat"])

PLATFORM_LABEL = {"telegram": "تلگرام", "bale": "بله"}


# ---------- agent settings ----------


class Faq(BaseModel):
    q: str = Field("", max_length=300)
    a: str = Field("", max_length=1000)


class AgentIn(BaseModel):
    enabled: bool | None = None
    assistant_name: str | None = Field(None, max_length=60)
    tone: str | None = None
    custom_instructions: str | None = Field(None, max_length=500)
    welcome_message: str | None = Field(None, max_length=1000)
    off_topic_message: str | None = Field(None, max_length=500)
    catalog_enabled: bool | None = None
    orders_enabled: bool | None = None
    payment: dict[str, Any] | None = None
    required_fields: dict[str, bool] | None = None
    faqs: list[Faq] | None = Field(None, max_length=40)
    rules: str | None = Field(None, max_length=3000)
    business_hours: dict[str, Any] | None = None
    handoff_minutes: int | None = Field(None, ge=5, le=1440)


def _agent_out(agent: ChatAgent, store: Store) -> dict[str, Any]:
    return {
        "enabled": agent.enabled, "assistant_name": agent.assistant_name, "tone": agent.tone,
        "custom_instructions": agent.custom_instructions, "welcome_message": agent.welcome_message,
        "off_topic_message": agent.off_topic_message, "catalog_enabled": agent.catalog_enabled,
        "orders_enabled": agent.orders_enabled, "payment": {"require_receipt": True, **(agent.payment or {})},
        "required_fields": {"name": True, "phone": True, "address": True, **(agent.required_fields or {})},
        "faqs": agent.faqs or [], "rules": agent.rules,
        "business_hours": {"enabled": False, "days": {}, "away_mode": "ai", "away_message": "", **(agent.business_hours or {})},
        "handoff_minutes": agent.handoff_minutes,
        "defaults": {"welcome_message": DEFAULT_WELCOME.replace("{store}", store.name),
                     "off_topic_message": DEFAULT_OFF_TOPIC.replace("{store}", store.name),
                     "away_message": DEFAULT_AWAY},
    }


@router.get("/stores/{store_id}/chat/agent")
async def get_agent_settings(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    agent = await get_agent(db, store_id)
    await db.commit()
    return _agent_out(agent, store)


PAYMENT_KEYS = ("card_number", "sheba", "holder", "link", "note", "require_receipt")


@router.put("/stores/{store_id}/chat/agent")
async def save_agent_settings(store_id: int, body: AgentIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    agent = await get_agent(db, store_id)
    data = body.model_dump(exclude_none=True)
    if "tone" in data and data["tone"] not in ("formal", "friendly", "casual"):
        raise HTTPException(422, "لحن نامعتبر است.")
    if "payment" in data:
        data["payment"] = {k: (bool(v) if k == "require_receipt" else str(v or "").strip()[:500])
                           for k, v in data["payment"].items() if k in PAYMENT_KEYS}
    if "faqs" in data:
        data["faqs"] = [{"q": f["q"].strip(), "a": f["a"].strip()} for f in data["faqs"] if f["q"].strip() and f["a"].strip()]
    if "business_hours" in data:
        bh = data["business_hours"]
        data["business_hours"] = {
            "enabled": bool(bh.get("enabled")),
            "days": {k: [[str(a)[:5], str(b)[:5]] for a, b in (v or [])[:3]] for k, v in (bh.get("days") or {}).items()
                     if k in ("sat", "sun", "mon", "tue", "wed", "thu", "fri")},
            "away_mode": "message" if bh.get("away_mode") == "message" else "ai",
            "away_message": str(bh.get("away_message") or "")[:500],
        }
    if data.get("enabled") and (problem := await route_problem("chat")):
        raise HTTPException(422, {"message": "دستیار بدون هوش مصنوعی روشن نمی‌شود: " + problem})
    if ("orders_enabled" in data or "payment" in data) and data.get("orders_enabled", agent.orders_enabled):
        pay = data.get("payment", agent.payment or {})
        if not any(pay.get(k) for k in ("card_number", "sheba", "link")):
            raise HTTPException(422, "برای سفارش‌گیری، حداقل شماره کارت، شبا یا لینک پرداخت را وارد کن.")
    for k, v in data.items():
        setattr(agent, k, v)
    await db.commit()
    return _agent_out(agent, store)


# ---------- connections ----------


def _bot_out(b: ChatBot) -> dict[str, Any]:
    return {
        "id": b.id, "platform": b.platform, "mode": b.mode, "status": b.status, "bot_username": b.bot_username,
        "bot_name": b.bot_name, "admin_connected": bool(b.admin_chat_id), "admin_name": b.admin_name,
        "business_user_name": b.business_user_name, "can_reply": b.can_reply, "last_update_at": b.last_update_at,
        "last_check_at": b.last_check_at, "last_error": b.last_error, "created_at": b.created_at,
    }


@router.get("/stores/{store_id}/chat/platforms")
async def chat_platforms(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    return [
        {"platform": "telegram", "label": "تلگرام", "modes": {"bot": True, "business": True}},
        {"platform": "bale", "label": "بله",
         "modes": {"bot": True, "business": False},
         "business_note": "بله در API رسمی ربات‌ها هنوز امکان اتصال به پیوی حساب شخصی را ندارد."},
    ]


@router.get("/stores/{store_id}/chat/bots")
async def list_bots(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(ChatBot).where(ChatBot.store_id == store_id, ChatBot.status != "disconnected")
                             .order_by(ChatBot.id))).scalars().all()
    return [_bot_out(b) for b in rows]


class BotIn(BaseModel):
    platform: str
    mode: str = "bot"
    bot_token: str = Field(min_length=10, max_length=200)


async def _token_in_use(db, token: str, platform: str, exclude_bot: int | None = None) -> str:
    """A chat bot can't share its token with a publishing channel or another chat connection: both read
    getUpdates and would take each other's messages."""
    h = tg.token_hash(token)
    q = select(ChatBot).where(ChatBot.token_hash == h, ChatBot.status != "disconnected")
    if exclude_bot:
        q = q.where(ChatBot.id != exclude_bot)
    if (await db.execute(q)).scalars().first():
        return "این ربات قبلاً برای دستیار چت وصل شده است."
    for ch in (await db.execute(select(Channel).where(Channel.platform == platform,
                                                      Channel.status != "disconnected"))).scalars():
        try:
            if decrypt_credentials(ch.credentials_encrypted).get("bot_token", "").strip() == token.strip():
                return ("این همان رباتی است که برای انتشار در کانال وصل کرده‌ای. برای چت یک ربات جدا بساز؛ "
                        "اگر یکی باشند پیام‌های همدیگر را می‌گیرند.")
        except Exception:  # noqa: BLE001
            continue
    return ""


@router.post("/stores/{store_id}/chat/bots")
async def connect_bot(store_id: int, body: BotIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    if body.platform not in tg.ADAPTERS:
        raise HTTPException(422, "پلتفرم پشتیبانی نمی‌شود.")
    if body.mode not in ("bot", "business"):
        raise HTTPException(422, "نوع اتصال نامعتبر است.")
    if body.mode == "business" and body.platform not in tg.BUSINESS_PLATFORMS:
        raise HTTPException(422, "بله هنوز امکان اتصال ربات به پیوی حساب شخصی را ندارد؛ از «بات مستقل» استفاده کن.")
    token = body.bot_token.strip()
    if problem := await _token_in_use(db, token, body.platform):
        raise HTTPException(422, problem)
    try:
        me = await tg.call(body.platform, "getMe", token=token)
    except PlatformError as exc:
        raise HTTPException(400, exc.message_fa)
    if body.mode == "business" and me.get("can_connect_to_business") is False:
        raise HTTPException(422, "Business Mode این ربات خاموش است. در ‎@BotFather › Bot Settings › Business Mode "
                                 "را روشن کن و دوباره امتحان کن.")
    bot = ChatBot(store_id=store_id, platform=body.platform, mode=body.mode,
                  status="waiting_business" if body.mode == "business" else "connected",
                  credentials_encrypted=encrypt_credentials({"bot_token": token}), token_hash=tg.token_hash(token),
                  bot_user_id=str(me.get("id", "")), bot_username=me.get("username", ""),
                  bot_name=me.get("first_name", ""), last_check_at=utcnow(), webhook_secret=rt.new_secret())
    db.add(bot)
    await db.flush()
    try:
        await rt.setup_transport(bot)
    except PlatformError as exc:
        await db.rollback()
        raise HTTPException(400, "تنظیم دریافت پیام ربات ناموفق بود: " + exc.message_fa)
    await db.commit()
    return _bot_out(bot)


async def _bot_for(db, user: User, bot_id: int) -> ChatBot:
    bot = await db.get(ChatBot, bot_id)
    if not bot or bot.status == "disconnected":
        raise HTTPException(404, "اتصال پیدا نشد.")
    await get_store_for(db, user, bot.store_id)
    return bot


@router.post("/chat/bots/{bot_id}/check")
async def check_bot(bot_id: int, user: CurrentUser, db: DB):
    bot = await _bot_for(db, user, bot_id)
    bot.last_check_at = utcnow()
    try:
        me = await tg.call(bot, "getMe")
        bot.bot_username, bot.bot_name = me.get("username", ""), me.get("first_name", "")
        if bot.mode == "business":
            if bot.business_connection_id:
                conn = await tg.call(bot, "getBusinessConnection", {"business_connection_id": bot.business_connection_id})
                await rt._business_connection(db, bot, conn)  # noqa: SLF001
            else:
                bot.status = "waiting_business"
        else:
            bot.status, bot.last_error = "connected", ""
        await rt.setup_transport(bot)
    except PlatformError as exc:
        bot.status = "error" if exc.code == "invalid_token" else bot.status
        bot.last_error = exc.message_fa
    await db.commit()
    return _bot_out(bot)


@router.post("/chat/bots/{bot_id}/admin-code")
async def admin_code(bot_id: int, user: CurrentUser, db: DB):
    bot = await _bot_for(db, user, bot_id)
    if bot.mode == "business":
        raise HTTPException(422, "در اتصال پیوی، خود حساب تلگرامت ادمین است و اعلان‌ها همان‌جا می‌آید.")
    bot.admin_link_code = f"{secrets.randbelow(900000) + 100000}"
    bot.admin_link_expires_at = utcnow() + timedelta(minutes=15)
    await db.commit()
    return {"code": bot.admin_link_code, "command": f"/admin {bot.admin_link_code}", "bot_username": bot.bot_username,
            "expires_at": bot.admin_link_expires_at}


@router.delete("/chat/bots/{bot_id}/admin")
async def unlink_admin(bot_id: int, user: CurrentUser, db: DB):
    bot = await _bot_for(db, user, bot_id)
    bot.admin_chat_id, bot.admin_name = "", ""
    await db.commit()
    return _bot_out(bot)


@router.delete("/chat/bots/{bot_id}")
async def disconnect_bot(bot_id: int, user: CurrentUser, db: DB):
    bot = await _bot_for(db, user, bot_id)
    try:
        if bot.platform and bot.credentials_encrypted:
            await tg.call(bot, "deleteWebhook", {})
    except PlatformError:
        pass
    bot.status, bot.credentials_encrypted, bot.token_hash = "disconnected", None, ""
    await db.commit()
    return {"ok": True}


@router.post("/chat/hook/{bot_id}/{secret}")
async def webhook(bot_id: int, secret: str, request: Request, db: DB, background: BackgroundTasks):
    bot = await db.get(ChatBot, bot_id)
    if not bot or not bot.webhook_secret or not secrets.compare_digest(secret, bot.webhook_secret):
        raise HTTPException(404, "not found")
    header = request.headers.get("x-telegram-bot-api-secret-token")
    if bot.platform == "telegram" and header is not None and not secrets.compare_digest(header, bot.webhook_secret):
        raise HTTPException(403, "bad secret")
    update = await request.json()
    background.add_task(rt.handle_update, bot_id, update)
    return {"ok": True}


# ---------- knowledge: FAQ from pasted text or a file ----------


@router.post("/stores/{store_id}/chat/faq/extract")
async def extract_faq(store_id: int, user: CurrentUser, db: DB, text: str = Form(""),
                      file: UploadFile | None = File(None)):
    """AI reads the seller's text/file and suggests Q&A pairs; the page shows them for review before saving."""
    await get_store_for(db, user, store_id)
    body = text.strip()
    if file is not None and file.filename:
        try:
            body = (body + "\n\n" + chat_faq.file_text(file.filename, await file.read())).strip()
        except chat_faq.FaqFileError as exc:
            raise HTTPException(422, str(exc))
    if len(body) < 20:
        raise HTTPException(422, "متن خیلی کوتاه است؛ متن سؤالات یا قوانین فروشگاه را بگذار.")
    agent = await get_agent(db, store_id)
    try:
        items, usage = await chat_faq.extract(body, store_id, agent.faqs or [])
    except AIError as exc:
        raise HTTPException(400, exc.message_fa)
    return {"items": items, "truncated": len(body) > chat_faq.MAX_CHARS, "usage": usage}


# ---------- knowledge: references ----------


class RefIn(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    keywords: str = Field("", max_length=300)
    text: str = Field("", max_length=1500)
    image_url: str = Field("", max_length=1000)


def _ref_out(r: ChatReference) -> dict[str, Any]:
    return {"id": r.id, "title": r.title, "keywords": r.keywords, "text": r.text, "image_url": r.image_url}


@router.get("/stores/{store_id}/chat/references")
async def list_refs(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(ChatReference).where(ChatReference.store_id == store_id)
                             .order_by(ChatReference.id))).scalars().all()
    return [_ref_out(r) for r in rows]


def _check_ref(body: RefIn, store_id: int) -> None:
    if not (body.text.strip() or body.image_url):
        raise HTTPException(422, "رفرنس باید عکس یا متن داشته باشد.")
    if body.image_url and not body.image_url.startswith(f"/media/{store_id}/"):
        raise HTTPException(422, "عکس رفرنس را از همین صفحه آپلود کن.")


@router.post("/stores/{store_id}/chat/references")
async def add_ref(store_id: int, body: RefIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    _check_ref(body, store_id)
    if await db.scalar(select(func.count()).select_from(ChatReference).where(ChatReference.store_id == store_id)) >= 30:
        raise HTTPException(422, "حداکثر ۳۰ رفرنس می‌توانی داشته باشی.")
    r = ChatReference(store_id=store_id, **body.model_dump())
    db.add(r)
    await db.commit()
    return _ref_out(r)


@router.put("/chat/references/{ref_id}")
async def update_ref(ref_id: int, body: RefIn, user: CurrentUser, db: DB):
    r = await db.get(ChatReference, ref_id)
    if not r:
        raise HTTPException(404, "رفرنس پیدا نشد.")
    await get_store_for(db, user, r.store_id)
    _check_ref(body, r.store_id)
    for k, v in body.model_dump().items():
        setattr(r, k, v)
    await db.commit()
    return _ref_out(r)


@router.delete("/chat/references/{ref_id}")
async def delete_ref(ref_id: int, user: CurrentUser, db: DB):
    r = await db.get(ChatReference, ref_id)
    if not r:
        raise HTTPException(404, "رفرنس پیدا نشد.")
    await get_store_for(db, user, r.store_id)
    await db.delete(r)
    await db.commit()
    return {"ok": True}


# ---------- test chat ----------


class TestIn(BaseModel):
    text: str = Field("", max_length=1000)
    reset: bool = False


def _msg_out(m: ChatMessage) -> dict[str, Any]:
    return {"id": m.id, "created_at": m.created_at, "role": m.role, "text": m.text, "image_url": m.image_url,
            "source": m.source, "intent": m.intent, "model": m.model, "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens, "cached_tokens": m.cached_tokens, "cost_toman": m.cost_toman,
            "latency_ms": m.latency_ms}


async def _test_conv(db, store_id: int, user: User) -> ChatConversation:
    conv = (await db.execute(select(ChatConversation).where(
        ChatConversation.store_id == store_id, ChatConversation.is_test,
        ChatConversation.external_chat_id == f"test-{user.id}"))).scalar_one_or_none()
    if not conv:
        conv = ChatConversation(store_id=store_id, bot_id=None, platform="test", mode="test", is_test=True,
                                external_chat_id=f"test-{user.id}", customer_name="تست فروشنده", state={})
        db.add(conv)
        await db.flush()
    return conv


@router.get("/stores/{store_id}/chat/test")
async def test_history(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    conv = await _test_conv(db, store_id, user)
    await db.commit()
    rows = (await db.execute(select(ChatMessage).where(ChatMessage.conversation_id == conv.id)
                             .order_by(ChatMessage.id))).scalars().all()
    return {"conversation_id": conv.id, "messages": [_msg_out(m) for m in rows], "state": conv.state}


@router.post("/stores/{store_id}/chat/test")
async def test_chat(store_id: int, body: TestIn, user: CurrentUser, db: DB):
    """The seller's simulator: same engine and logging as a real chat, marked as a test."""
    await get_store_for(db, user, store_id)
    conv = await _test_conv(db, store_id, user)
    if body.reset:
        conv.is_test = True
        conv.external_chat_id = f"test-{user.id}-old-{conv.id}"
        await db.commit()
        conv = await _test_conv(db, store_id, user)
    if not body.text.strip():
        await db.commit()
        return {"conversation_id": conv.id, "messages": [], "state": conv.state}
    customer = rt.add_message(db, conv, role="customer", text=body.text.strip())
    await db.commit()
    replies = await rt.answer(conv.id, send=False)
    await db.refresh(conv)
    return {"conversation_id": conv.id, "messages": [_msg_out(customer)] + [_msg_out(m) for m in replies],
            "state": conv.state}


# ---------- conversations ----------


def _conv_out(c: ChatConversation, orders: int = 0, bot: ChatBot | None = None) -> dict[str, Any]:
    return {"id": c.id, "platform": c.platform, "mode": c.mode, "customer_name": c.customer_name,
            "customer_username": c.customer_username, "status": c.status, "ai_paused_until": c.ai_paused_until,
            "message_count": c.message_count, "ai_message_count": c.ai_message_count, "llm_calls": c.llm_calls,
            "tokens_in": c.tokens_in, "tokens_out": c.tokens_out, "tokens_cached": c.tokens_cached,
            "cost_toman": c.cost_toman, "last_message_at": c.last_message_at, "last_message_text": c.last_message_text,
            "seller_unread": c.seller_unread, "orders": orders, "created_at": c.created_at,
            "bot_username": bot.bot_username if bot else ""}


@router.get("/stores/{store_id}/chat/conversations")
async def list_conversations(store_id: int, user: CurrentUser, db: DB, platform: str = "", status: str = "",
                             has_order: bool = False, q: str = "", limit: int = 50, offset: int = 0):
    await get_store_for(db, user, store_id)
    orders = (select(AiOrder.conversation_id, func.count().label("n")).where(AiOrder.store_id == store_id)
              .group_by(AiOrder.conversation_id).subquery())
    stmt = (select(ChatConversation, func.coalesce(orders.c.n, 0))
            .outerjoin(orders, orders.c.conversation_id == ChatConversation.id)
            .where(ChatConversation.store_id == store_id, ~ChatConversation.is_test))
    if platform:
        stmt = stmt.where(ChatConversation.platform == platform)
    if status:
        stmt = stmt.where(ChatConversation.status == status)
    if has_order:
        stmt = stmt.where(orders.c.n > 0)
    if q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(ChatConversation.customer_name.ilike(like) | ChatConversation.customer_username.ilike(like)
                          | ChatConversation.last_message_text.ilike(like))
    stmt = stmt.order_by(ChatConversation.last_message_at.desc().nullslast(), ChatConversation.id.desc())
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = (await db.execute(stmt.limit(min(limit, 100)).offset(offset))).all()
    bots = {b.id: b for b in (await db.execute(select(ChatBot).where(ChatBot.store_id == store_id))).scalars()}
    needs = await db.scalar(select(func.count()).select_from(ChatConversation).where(
        ChatConversation.store_id == store_id, ChatConversation.status == "needs_human", ~ChatConversation.is_test))
    return {"total": total, "needs_human": needs,
            "items": [_conv_out(c, n, bots.get(c.bot_id)) for c, n in rows]}


async def _conv_for(db, user: User, conv_id: int) -> ChatConversation:
    c = await db.get(ChatConversation, conv_id)
    if not c:
        raise HTTPException(404, "گفتگو پیدا نشد.")
    await get_store_for(db, user, c.store_id)
    return c


def _order_out(o: AiOrder) -> dict[str, Any]:
    return {"id": o.id, "conversation_id": o.conversation_id, "platform": o.platform, "customer_name": o.customer_name,
            "customer_phone": o.customer_phone, "customer_address": o.customer_address,
            "customer_username": o.customer_username, "items": o.items, "total_toman": o.total_toman, "note": o.note,
            "status": o.status, "payment_ref": o.payment_ref, "receipt_url": o.receipt_url,
            "receipt_text": o.receipt_text, "seller_note": o.seller_note, "is_test": o.is_test,
            "created_at": o.created_at, "updated_at": o.updated_at}


@router.get("/chat/conversations/{conv_id}")
async def get_conversation(conv_id: int, user: CurrentUser, db: DB):
    c = await _conv_for(db, user, conv_id)
    c.seller_unread = False
    msgs = (await db.execute(select(ChatMessage).where(ChatMessage.conversation_id == c.id)
                             .order_by(ChatMessage.id).limit(500))).scalars().all()
    orders = (await db.execute(select(AiOrder).where(AiOrder.conversation_id == c.id).order_by(AiOrder.id.desc()))).scalars().all()
    bot = await db.get(ChatBot, c.bot_id) if c.bot_id else None
    await db.commit()
    return {"conversation": _conv_out(c, len(orders), bot), "messages": [_msg_out(m) for m in msgs],
            "orders": [_order_out(o) for o in orders], "state": c.state}


class SellerMsgIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@router.post("/chat/conversations/{conv_id}/messages")
async def seller_message(conv_id: int, body: SellerMsgIn, user: CurrentUser, db: DB):
    c = await _conv_for(db, user, conv_id)
    try:
        row = await rt.seller_send(db, c, body.text.strip())
    except PlatformError as exc:
        raise HTTPException(400, exc.message_fa)
    await db.commit()
    return _msg_out(row)


@router.post("/chat/conversations/{conv_id}/resume")
async def resume_ai(conv_id: int, user: CurrentUser, db: DB):
    c = await _conv_for(db, user, conv_id)
    c.status, c.ai_paused_until = "ai", None
    state = dict(c.state or {})
    state.pop("limit_notified", None)
    state.pop("slowed", None)
    c.state = state
    await db.commit()
    return _conv_out(c)


@router.post("/chat/conversations/{conv_id}/pause")
async def pause(conv_id: int, user: CurrentUser, db: DB):
    c = await _conv_for(db, user, conv_id)
    await rt.pause_ai(db, c)
    await db.commit()
    return _conv_out(c)


# ---------- AI orders ----------


@router.get("/stores/{store_id}/ai-orders")
async def list_orders(store_id: int, user: CurrentUser, db: DB, status: str = "", include_test: bool = False,
                      limit: int = 50, offset: int = 0):
    await get_store_for(db, user, store_id)
    stmt = select(AiOrder).where(AiOrder.store_id == store_id)
    if not include_test:
        stmt = stmt.where(~AiOrder.is_test)
    if status:
        stmt = stmt.where(AiOrder.status == status)
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = (await db.execute(stmt.order_by(AiOrder.id.desc()).limit(min(limit, 100)).offset(offset))).scalars().all()
    counts = dict((await db.execute(select(AiOrder.status, func.count()).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test).group_by(AiOrder.status))).all())
    return {"total": total, "counts": counts, "items": [_order_out(o) for o in rows]}


class OrderPatch(BaseModel):
    status: str | None = None
    seller_note: str | None = Field(None, max_length=1000)
    notify_customer: bool = True


ORDER_STATUS_TEXT = {
    "confirmed": "✅ پرداخت سفارش #{id} تأیید شد و سفارشت در حال آماده‌سازیه. ممنون از خریدت 🌸",
    "rejected": "سفارش #{id} تأیید نشد. برای پیگیری همین‌جا پیام بده 🙏",
    "cancelled": "سفارش #{id} لغو شد.",
}


@router.patch("/ai-orders/{order_id}")
async def update_order(order_id: int, body: OrderPatch, user: CurrentUser, db: DB):
    o = await db.get(AiOrder, order_id)
    if not o:
        raise HTTPException(404, "سفارش پیدا نشد.")
    await get_store_for(db, user, o.store_id)
    if body.seller_note is not None:
        o.seller_note = body.seller_note
    if body.status and body.status != o.status:
        if body.status not in ("awaiting_payment", "payment_submitted", "confirmed", "rejected", "cancelled"):
            raise HTTPException(422, "وضعیت نامعتبر است.")
        if body.status == "confirmed" and o.status != "confirmed":
            for it in o.items or []:  # stock goes down only when the seller confirms
                p = await db.get(Product, it.get("product_id") or 0) if it.get("product_id") else None
                if p and p.store_id == o.store_id:
                    p.stock_qty = max(0, p.stock_qty - int(it.get("qty") or 0))
        o.status = body.status
        conv = await db.get(ChatConversation, o.conversation_id)
        if conv:
            state = dict(conv.state or {})
            if state.get("order_id") == o.id and body.status in ("confirmed", "rejected", "cancelled"):
                state.pop("awaiting", None)
                conv.state = state
            if body.notify_customer and body.status in ORDER_STATUS_TEXT and not conv.is_test:
                try:
                    await rt.seller_send(db, conv, ORDER_STATUS_TEXT[body.status].format(id=o.id))
                    conv.status, conv.ai_paused_until = "ai", None  # a status note shouldn't silence the assistant
                except PlatformError:
                    pass
    await db.commit()
    return _order_out(o)


# ---------- overview & usage ----------


def _month_start() -> datetime:
    now = utcnow()
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


@router.get("/stores/{store_id}/chat/overview")
async def overview(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    agent = await get_agent(db, store_id)
    await db.commit()
    bots = (await db.execute(select(ChatBot).where(ChatBot.store_id == store_id, ChatBot.status != "disconnected"))).scalars().all()
    today = datetime.combine(utcnow().date(), time.min, tzinfo=timezone.utc)
    real = [ChatMessage.store_id == store_id, ~ChatMessage.is_test]
    msgs_today = await db.scalar(select(func.count()).select_from(ChatMessage).where(
        *real, ChatMessage.created_at >= today, ChatMessage.role == "customer")) or 0
    ai_today = await db.scalar(select(func.count()).select_from(ChatMessage).where(
        *real, ChatMessage.created_at >= today, ChatMessage.role == "ai")) or 0
    pending_orders = await db.scalar(select(func.count()).select_from(AiOrder).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.status == "payment_submitted")) or 0
    open_orders = await db.scalar(select(func.count()).select_from(AiOrder).where(
        AiOrder.store_id == store_id, ~AiOrder.is_test, AiOrder.status == "awaiting_payment")) or 0
    needs_human = await db.scalar(select(func.count()).select_from(ChatConversation).where(
        ChatConversation.store_id == store_id, ~ChatConversation.is_test, ChatConversation.status == "needs_human")) or 0
    tokens = (await db.execute(select(func.coalesce(func.sum(ChatMessage.input_tokens + ChatMessage.output_tokens), 0),
                                      func.coalesce(func.sum(ChatMessage.cost_toman), 0))
                               .where(*real, ChatMessage.created_at >= _month_start()))).one()
    refs = await db.scalar(select(func.count()).select_from(ChatReference).where(ChatReference.store_id == store_id)) or 0
    ai_problem = await route_problem("chat")
    setup = [
        {"key": "ai", "done": not ai_problem, "label": "هوش مصنوعی توسط ادمین Sello تنظیم شده"},
        {"key": "connect", "done": any(b.status == "connected" for b in bots), "label": "اتصال تلگرام یا بله"},
        {"key": "welcome", "done": bool(agent.welcome_message), "label": "پیام خوشامد و لحن"},
        {"key": "knowledge", "done": bool(agent.faqs or agent.rules or refs), "label": "سؤالات متداول، قوانین یا رفرنس"},
        {"key": "orders", "done": agent.orders_enabled, "label": "سفارش‌گیری و اطلاعات پرداخت"},
        {"key": "test", "done": bool(await db.scalar(select(ChatConversation.id).where(
            ChatConversation.store_id == store_id, ChatConversation.is_test, ChatConversation.message_count > 0).limit(1))),
         "label": "یک بار تست چت"},
    ]
    return {
        "enabled": agent.enabled, "ai_ready": not ai_problem, "ai_problem": ai_problem if user.role == "admin" else "",
        "bots": [_bot_out(b) for b in bots], "setup": setup,
        "today": {"customer_messages": msgs_today, "ai_replies": ai_today},
        "orders": {"payment_submitted": pending_orders, "awaiting_payment": open_orders},
        "needs_human": needs_human,
        "month": {"ai_messages": await month_ai_messages(db, store_id), "limit": await chat_limit(db, store),
                  "tokens": int(tokens[0]), "cost_toman": int(tokens[1])},
    }


@router.get("/stores/{store_id}/chat/usage")
async def usage(store_id: int, user: CurrentUser, db: DB, days: int = 30, include_test: bool = False):
    await get_store_for(db, user, store_id)
    days = max(1, min(days, 180))
    since = datetime.combine(utcnow().date() - timedelta(days=days - 1), time.min, tzinfo=timezone.utc)
    base = [ChatMessage.store_id == store_id, ChatMessage.created_at >= since]
    if not include_test:
        base.append(~ChatMessage.is_test)
    is_llm = ChatMessage.model != ""
    agg = (func.count(),
           func.sum(case((ChatMessage.role == "customer", 1), else_=0)),
           func.sum(case((ChatMessage.role == "ai", 1), else_=0)),
           func.sum(case((is_llm, 1), else_=0)),
           func.coalesce(func.sum(ChatMessage.input_tokens), 0), func.coalesce(func.sum(ChatMessage.output_tokens), 0),
           func.coalesce(func.sum(ChatMessage.cached_tokens), 0), func.coalesce(func.sum(ChatMessage.cost_toman), 0))
    total = (await db.execute(select(*agg).where(*base))).one()
    convs = await db.scalar(select(func.count(func.distinct(ChatMessage.conversation_id))).where(*base)) or 0
    day_col = func.date(ChatMessage.created_at)
    series = [{"day": str(d), "messages": int(n or 0), "customer": int(c or 0), "ai": int(a or 0), "llm": int(llm or 0),
               "tokens": int((tin or 0) + (tout or 0)), "cached": int(cached or 0), "cost_toman": int(cost or 0)}
              for d, n, c, a, llm, tin, tout, cached, cost in
              (await db.execute(select(day_col, *agg).where(*base).group_by(day_col).order_by(day_col))).all()]
    by_source = dict((await db.execute(select(ChatMessage.source, func.count()).where(
        *base, ChatMessage.role == "ai").group_by(ChatMessage.source))).all())
    conv_platform = (select(ChatConversation.platform, func.count(func.distinct(ChatMessage.conversation_id)),
                            func.count(), func.coalesce(func.sum(ChatMessage.input_tokens + ChatMessage.output_tokens), 0))
                     .join(ChatConversation, ChatConversation.id == ChatMessage.conversation_id)
                     .where(*base).group_by(ChatConversation.platform))
    by_platform = [{"platform": p, "conversations": c, "messages": m, "tokens": int(t)}
                   for p, c, m, t in (await db.execute(conv_platform)).all()]
    recent = (await db.execute(select(ChatMessage, ChatConversation.customer_name, ChatConversation.platform)
                               .join(ChatConversation, ChatConversation.id == ChatMessage.conversation_id)
                               .where(*base, is_llm).order_by(ChatMessage.id.desc()).limit(40))).all()
    orders = await db.scalar(select(func.count()).select_from(AiOrder).where(
        AiOrder.store_id == store_id, AiOrder.created_at >= since, *([] if include_test else [~AiOrder.is_test]))) or 0
    _, customer, ai_n, llm, tin, tout, cached, cost = total
    return {
        "days": days,
        "totals": {"conversations": convs, "customer_messages": int(customer or 0), "ai_replies": int(ai_n or 0),
                   "llm_calls": int(llm or 0), "without_llm": int((ai_n or 0) - (llm or 0)),
                   "input_tokens": int(tin), "output_tokens": int(tout), "cached_tokens": int(cached),
                   "cost_toman": int(cost), "orders": orders,
                   "avg_tokens_per_call": round((tin + tout) / llm) if llm else 0},
        "series": series, "by_source": by_source, "by_platform": by_platform,
        "recent": [{**_msg_out(m), "customer_name": name, "platform": platform, "conversation_id": m.conversation_id}
                   for m, name, platform in recent],
    }


@router.get("/admin/chat/usage")
async def admin_usage(admin: AdminUser, db: DB, days: int = 30):
    days = max(1, min(days, 180))
    since = datetime.combine(utcnow().date() - timedelta(days=days - 1), time.min, tzinfo=timezone.utc)
    base = [ChatMessage.created_at >= since, ~ChatMessage.is_test]
    agg = (func.count(func.distinct(ChatMessage.conversation_id)), func.count(),
           func.sum(case((ChatMessage.model != "", 1), else_=0)),
           func.coalesce(func.sum(ChatMessage.input_tokens), 0), func.coalesce(func.sum(ChatMessage.output_tokens), 0),
           func.coalesce(func.sum(ChatMessage.cached_tokens), 0), func.coalesce(func.sum(ChatMessage.cost_toman), 0))
    t = (await db.execute(select(*agg).where(*base))).one()
    rows = (await db.execute(select(ChatMessage.store_id, Store.name, User.phone, *agg)
                             .join(Store, Store.id == ChatMessage.store_id).join(User, User.id == Store.owner_id)
                             .where(*base).group_by(ChatMessage.store_id, Store.name, User.phone)
                             .order_by(func.count().desc()).limit(100))).all()
    orders = dict((await db.execute(select(AiOrder.store_id, func.count()).where(
        AiOrder.created_at >= since, ~AiOrder.is_test).group_by(AiOrder.store_id))).all())
    bots = dict((await db.execute(select(ChatBot.store_id, func.count()).where(
        ChatBot.status != "disconnected").group_by(ChatBot.store_id))).all())
    day_col = func.date(ChatMessage.created_at)
    series = [{"day": str(d), "conversations": c, "messages": n, "llm": int(llm or 0), "tokens": int(tin + tout),
               "cost_toman": int(cost)}
              for d, c, n, llm, tin, tout, cached, cost in
              (await db.execute(select(day_col, *agg).where(*base).group_by(day_col).order_by(day_col))).all()]
    return {
        "days": days,
        "totals": {"conversations": t[0], "messages": t[1], "llm_calls": int(t[2] or 0), "input_tokens": int(t[3]),
                   "output_tokens": int(t[4]), "cached_tokens": int(t[5]), "cost_toman": int(t[6]),
                   "orders": sum(orders.values()), "active_bots": sum(bots.values())},
        "series": series,
        "stores": [{"store_id": sid, "store_name": name, "owner_phone": phone, "conversations": c, "messages": n,
                    "llm_calls": int(llm or 0), "tokens": int(tin + tout), "cached_tokens": int(cached),
                    "cost_toman": int(cost), "orders": orders.get(sid, 0), "bots": bots.get(sid, 0)}
                   for sid, name, phone, c, n, llm, tin, tout, cached, cost in rows],
    }
