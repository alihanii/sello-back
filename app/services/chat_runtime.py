"""Running the assistant: updates in (long polling or webhook), messages saved, bursts merged, replies sent.

Long polling is the default because it works on a laptop without a public address; with
SELLO_CHAT_TRANSPORT=webhook each bot gets a webhook instead (needed when more than one server runs)."""

import asyncio
import logging
import secrets
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import SessionLocal, utcnow
from app.models import ChatBot, ChatConversation, ChatMessage, Store
from app.platforms.base import PlatformError
from app.services import chat_messenger as tg
from app.services.chat_engine import Reply, _aware, get_agent, run_turn
from app.services.media import read_local, save_bytes
from app.services.notify import notify

log = logging.getLogger("sello.chat")

_timers: dict[int, asyncio.Task] = {}
_locks: dict[int, asyncio.Lock] = {}


def debounce_seconds() -> float:
    return settings.chat_debounce_seconds


# ---------- incoming ----------


async def handle_update(bot_id: int, update: dict[str, Any]) -> None:
    async with SessionLocal() as db:
        bot = await db.get(ChatBot, bot_id)
        if not bot or bot.status == "disconnected":
            return
        bot.last_update_at = utcnow()
        if conn := update.get("business_connection"):
            await _business_connection(db, bot, conn)
            await db.commit()
            return
        msg = update.get("business_message") or update.get("message")
        if not msg:
            await db.commit()
            return
        business = "business_message" in update
        inbound = tg.parse_message(bot, msg, business)
        if not inbound or (business and bot.mode != "business") or (not business and bot.mode == "business"):
            await db.commit()
            return
        if not business and await _admin_command(db, bot, inbound):
            await db.commit()
            return
        if not business and bot.admin_chat_id and inbound.chat_id == bot.admin_chat_id:
            await db.commit()  # the seller's own chat with the bot is for notifications, not a customer
            return
        conv_id = await receive(db, bot, inbound)
    if conv_id:
        schedule(conv_id)


async def _business_connection(db: AsyncSession, bot: ChatBot, conn: dict[str, Any]) -> None:
    user = conn.get("user") or {}
    enabled = conn.get("is_enabled", True)
    bot.business_connection_id = conn.get("id", "") if enabled else ""
    bot.business_user_id = str(user.get("id", ""))
    bot.business_user_name = tg._name(user) or user.get("username", "")  # noqa: SLF001
    bot.can_reply = enabled and tg.business_rights(conn)
    if not enabled:
        bot.status, bot.last_error = "waiting_business", "اتصال ربات به حساب تلگرام برداشته شد."
    elif not bot.can_reply:
        bot.status, bot.last_error = "waiting_business", "ربات اجازهٔ پاسخ دادن ندارد؛ در تنظیمات Telegram Business گزینهٔ پاسخ به پیام‌ها را روشن کن."
    else:
        bot.status, bot.last_error = "connected", ""
        bot.admin_chat_id = str(conn.get("user_chat_id") or user.get("id") or "")
        bot.admin_name = bot.business_user_name


async def _admin_command(db: AsyncSession, bot: ChatBot, inbound: tg.Inbound) -> bool:
    """`/admin 1234` from the seller links their chat with the bot, for order and handoff notifications."""
    text = inbound.text.strip()
    if not text.lower().startswith("/admin"):
        return False
    code = text[6:].strip()
    expires = _aware(bot.admin_link_expires_at)
    ok = bool(bot.admin_link_code) and code == bot.admin_link_code and expires and expires > utcnow()
    if ok:
        bot.admin_chat_id, bot.admin_name = inbound.chat_id, inbound.name or inbound.username
        bot.admin_link_code, bot.admin_link_expires_at = "", None
    try:
        await tg.send_text(bot, inbound.chat_id, "✅ این چت به‌عنوان ادمین فروشگاه وصل شد؛ سفارش‌ها و پیام‌های مهم اینجا می‌آید."
                           if ok else "کد ادمین درست نیست یا منقضی شده؛ از پنل Sello کد تازه بگیر.")
    except PlatformError:
        pass
    return True


async def receive(db: AsyncSession, bot: ChatBot, inbound: tg.Inbound) -> int | None:
    """Saves the message; returns the conversation to answer, or None when nothing should be answered."""
    conv = (await db.execute(select(ChatConversation).where(
        ChatConversation.bot_id == bot.id, ChatConversation.external_chat_id == inbound.chat_id,
        ChatConversation.business_connection_id == inbound.business_connection_id))).scalar_one_or_none()
    if not conv:
        if inbound.from_owner:
            return None  # the seller wrote first to someone; nothing to track until the customer answers
        conv = ChatConversation(store_id=bot.store_id, bot_id=bot.id, platform=bot.platform, mode=bot.mode,
                                external_chat_id=inbound.chat_id, business_connection_id=inbound.business_connection_id,
                                customer_name=inbound.name, customer_username=inbound.username, state={})
        db.add(conv)
        await db.flush()
    if inbound.message_id and await db.scalar(select(ChatMessage.id).where(
            ChatMessage.conversation_id == conv.id, ChatMessage.external_id == inbound.message_id,
            ChatMessage.role.in_(["customer", "seller"]))):
        return None  # the same update delivered twice
    image_url = ""
    if inbound.photo_file_id:
        try:
            data, name = await tg.download(bot, inbound.photo_file_id)
            image_url = save_bytes(bot.store_id, data, name)
        except (PlatformError, Exception):  # noqa: BLE001 - a lost photo must not drop the message
            log.warning("chat photo download failed", exc_info=True)
    role = "seller" if inbound.from_owner else "customer"
    text = inbound.text or ("[عکس]" if inbound.photo_file_id and not image_url else "")
    add_message(db, conv, role=role, text=text, image_url=image_url, external_id=inbound.message_id)
    if inbound.from_owner:
        await pause_ai(db, conv)
        await db.commit()
        return None
    if inbound.name and not conv.customer_name:
        conv.customer_name = inbound.name
    conv.seller_unread = True
    await db.commit()
    return conv.id


def add_message(db: AsyncSession, conv: ChatConversation, *, role: str, text: str = "", image_url: str = "",
                source: str = "", intent: str = "", usage: dict[str, Any] | None = None,
                external_id: str = "") -> ChatMessage:
    usage = usage or {}
    m = ChatMessage(created_at=utcnow(), conversation_id=conv.id, store_id=conv.store_id, role=role, text=text,
                    image_url=image_url, source=source, intent=intent, model=usage.get("model", ""),
                    input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                    cached_tokens=usage.get("cached_tokens", 0), cost_toman=usage.get("cost_toman", 0),
                    latency_ms=usage.get("latency_ms", 0), external_id=external_id, is_test=conv.is_test)
    db.add(m)
    conv.message_count = (conv.message_count or 0) + 1
    conv.last_message_at = m.created_at
    conv.last_message_text = (text or ("[عکس]" if image_url else ""))[:300]
    if role == "ai":
        conv.ai_message_count = (conv.ai_message_count or 0) + 1
    if usage.get("model"):
        conv.llm_calls = (conv.llm_calls or 0) + 1
        conv.tokens_in = (conv.tokens_in or 0) + usage.get("input_tokens", 0)
        conv.tokens_out = (conv.tokens_out or 0) + usage.get("output_tokens", 0)
        conv.tokens_cached = (conv.tokens_cached or 0) + usage.get("cached_tokens", 0)
        conv.cost_toman = (conv.cost_toman or 0) + usage.get("cost_toman", 0)
    return m


async def pause_ai(db: AsyncSession, conv: ChatConversation) -> None:
    agent = await get_agent(db, conv.store_id)
    conv.status = "human"
    conv.ai_paused_until = utcnow() + timedelta(minutes=max(1, agent.handoff_minutes or 60))


# ---------- answering ----------


def schedule(conv_id: int) -> None:
    """Answer after a short quiet period, so a question typed as three messages costs one model call."""
    if old := _timers.get(conv_id):
        old.cancel()

    async def later() -> None:
        try:
            await asyncio.sleep(debounce_seconds())
        except asyncio.CancelledError:
            return
        _timers.pop(conv_id, None)
        try:
            await answer(conv_id)
        except Exception:  # noqa: BLE001
            log.exception("chat turn failed for conversation %s", conv_id)

    _timers[conv_id] = asyncio.create_task(later())


async def answer(conv_id: int, *, send: bool = True) -> list[ChatMessage]:
    """Runs one turn for the customer's unanswered messages; returns the replies saved."""
    lock = _locks.setdefault(conv_id, asyncio.Lock())
    async with lock, SessionLocal() as db:
        conv = await db.get(ChatConversation, conv_id)
        if not conv:
            return []
        agent = await get_agent(db, conv.store_id)
        store = await db.get(Store, conv.store_id)
        if not conv.is_test and not agent.enabled:
            await db.commit()
            return []
        paused = _aware(conv.ai_paused_until)
        if conv.status in ("human", "needs_human") and not conv.is_test:
            if paused and paused > utcnow():
                return []  # the seller has it (or was asked to); the assistant stays quiet for a while
            conv.status = "ai"
        last_reply = await db.scalar(select(ChatMessage.id).where(
            ChatMessage.conversation_id == conv.id, ChatMessage.role.in_(["ai", "seller"]))
            .order_by(ChatMessage.id.desc()).limit(1)) or 0
        pending = (await db.execute(select(ChatMessage).where(
            ChatMessage.conversation_id == conv.id, ChatMessage.role == "customer", ChatMessage.id > last_reply)
            .order_by(ChatMessage.id))).scalars().all()
        if not pending:
            return []
        result = await run_turn(db, conv, store, agent, list(pending))
        if conv.is_test and result.error and result.replies:
            # Only the seller sees the test chat: say why the model failed instead of a vague "colleague will answer".
            result.replies[-1].text += f"\n\n⚠️ (فقط در چت تست) خطای هوش مصنوعی: {result.error}"
        if conv.status == "needs_human":
            # Waiting for the seller: stay quiet instead of repeating "an admin will answer", for a while.
            conv.ai_paused_until = utcnow() + timedelta(minutes=max(1, agent.handoff_minutes or 60))
        saved: list[ChatMessage] = []
        for r in result.replies:
            saved.append(add_message(db, conv, role="ai", text=r.text, image_url=r.image_url, source=r.source,
                                     intent=r.intent, usage=r.usage))
        if result.notify_admin:
            for note in result.notify_admin:
                await notify(db, conv.store_id, "chat", note[:200], "", link=_conv_link(conv))
        await db.commit()
        if send and not conv.is_test and conv.bot_id:
            bot = await db.get(ChatBot, conv.bot_id)
            await deliver(db, bot, conv, result.replies, saved)
            if result.notify_admin:
                await notify_admin_chat(bot, "\n".join(result.notify_admin), conv)
            await db.commit()
        return saved


def _conv_link(conv: ChatConversation) -> str:
    return f"/app/s/{conv.store_id}/chat?tab=conversations&c={conv.id}"


async def deliver(db: AsyncSession, bot: ChatBot | None, conv: ChatConversation, replies: list[Reply],
                  saved: list[ChatMessage]) -> None:
    if not bot or bot.status == "disconnected":
        return
    biz = conv.business_connection_id if conv.mode == "business" else ""
    for reply, row in zip(replies, saved):
        try:
            if reply.image_url and (local := read_local(reply.image_url)):
                data, mime = local
                row.external_id = await tg.send_photo(bot, conv.external_chat_id, data, "ref" + _ext(mime),
                                                      reply.text, biz)
            elif reply.text:
                row.external_id = await tg.send_text(bot, conv.external_chat_id, reply.text, biz)
        except PlatformError as exc:
            bot.last_error = exc.message_fa
            log.warning("chat send failed: %s", exc.message_fa)
            break


def _ext(mime: str) -> str:
    return {"image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".jpg")


async def notify_admin_chat(bot: ChatBot | None, text: str, conv: ChatConversation | None = None) -> None:
    if not bot or not bot.admin_chat_id:
        return
    link = f"\n{settings.frontend_base_url}{_conv_link(conv)}" if conv else ""
    try:
        await tg.send_text(bot, bot.admin_chat_id, text + link)
    except PlatformError:
        log.warning("admin chat notify failed", exc_info=True)


async def seller_send(db: AsyncSession, conv: ChatConversation, text: str) -> ChatMessage:
    """The seller answers from the panel: sent through the bot, and the assistant steps back for a while."""
    row = add_message(db, conv, role="seller", text=text)
    await pause_ai(db, conv)
    conv.seller_unread = False
    if conv.bot_id and not conv.is_test:
        bot = await db.get(ChatBot, conv.bot_id)
        if not bot or bot.status == "disconnected":
            raise PlatformError("اتصال این گفتگو قطع است؛ پیام فرستاده نشد.", "bot_disconnected")
        biz = conv.business_connection_id if conv.mode == "business" else ""
        row.external_id = await tg.send_text(bot, conv.external_chat_id, text, biz)
    return row


# ---------- transport ----------


def new_secret() -> str:
    return secrets.token_urlsafe(24)


async def setup_transport(bot: ChatBot) -> None:
    """Webhook mode registers our URL; polling mode removes any webhook (getUpdates fails while one is set)."""
    if settings.chat_transport == "webhook":
        if not bot.webhook_secret:
            bot.webhook_secret = new_secret()
        data: dict[str, Any] = {"url": f"{settings.public_base_url.rstrip('/')}/api/v1/chat/hook/{bot.id}/{bot.webhook_secret}",
                                "allowed_updates": tg.ALLOWED_UPDATES}
        if bot.platform == "telegram":
            data["secret_token"] = bot.webhook_secret
        await tg.call(bot, "setWebhook", data)
    else:
        try:
            await tg.call(bot, "deleteWebhook", {"drop_pending_updates": False})
        except PlatformError:
            pass  # Bale returns an error when there is no webhook to delete


async def _poll_bot(bot_id: int, stop: asyncio.Event) -> None:
    backoff = 2.0
    while not stop.is_set():
        async with SessionLocal() as db:
            bot = await db.get(ChatBot, bot_id)
            if not bot or bot.status == "disconnected" or not bot.credentials_encrypted:
                return
            offset = bot.update_offset or 0
        try:
            updates = await tg.get_updates(bot, offset)
            backoff = 2.0
        except PlatformError as exc:
            async with SessionLocal() as db:
                row = await db.get(ChatBot, bot_id)
                if row:
                    row.last_error = exc.message_fa
                    if exc.code == "invalid_token":
                        row.status = "error"
                    await db.commit()
            if exc.code == "invalid_token":
                return
            await _sleep(stop, backoff)
            backoff = min(backoff * 2, 60)
            continue
        except Exception:  # noqa: BLE001 - network hiccups; keep polling
            await _sleep(stop, backoff)
            backoff = min(backoff * 2, 60)
            continue
        for upd in updates:
            try:
                await handle_update(bot_id, upd)
            except Exception:  # noqa: BLE001
                log.exception("chat update failed")
            async with SessionLocal() as db:
                row = await db.get(ChatBot, bot_id)
                if row:
                    row.update_offset = max(row.update_offset or 0, int(upd.get("update_id", 0)) + 1)
                    await db.commit()


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def poller_loop(stop: asyncio.Event) -> None:
    """One long-polling task per connected bot; picks up new bots and drops removed ones every few seconds."""
    tasks: dict[int, asyncio.Task] = {}
    while not stop.is_set():
        try:
            async with SessionLocal() as db:
                ids = set((await db.execute(select(ChatBot.id).where(
                    ChatBot.status.in_(["connected", "waiting_business"]),
                    ChatBot.credentials_encrypted.is_not(None)))).scalars())
            for bid in ids - tasks.keys():
                tasks[bid] = asyncio.create_task(_poll_bot(bid, stop))
            for bid, t in list(tasks.items()):
                if t.done():
                    tasks.pop(bid)
        except Exception:  # noqa: BLE001
            log.exception("chat poller loop error")
        await _sleep(stop, 5)
    for t in tasks.values():
        t.cancel()
