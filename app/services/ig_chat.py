"""Instagram DMs and comments answered by the chat assistant, for accounts connected with username + password.

The seller turns it on per Instagram channel (DMs, comments, or both). A ChatBot row (platform "instagram",
channel_id = the channel) ties the account to the chat assistant, so its conversations show in the chat pages like
Telegram's. Instagram sends no webhooks to the app API, so the inbox is read on a slow, jittered timer through the
account's own session and fixed proxy (instagram_private), never fast enough to look like a bot:
- DMs: the inbox and message requests every 75 to 150 seconds; requests with a new message are accepted;
- comments: the account's recent published posts every 6 to 10 minutes; each new comment gets a short public reply
  ("we sent you a DM") and the question itself goes to the assistant, which answers in the commenter's DM.
A message the seller types in the Instagram app pauses the assistant for that chat, as in Telegram Business."""

import asyncio
import logging
import random
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal, utcnow
from app.models import Channel, ChannelListing, ChatBot, ChatConversation, ChatMessage, Product
from app.platforms.base import PlatformError
from app.platforms.instagram_private import private
from app.services import proxy_pool

log = logging.getLogger("sello.ig_chat")

DM_EVERY = (75, 150)  # seconds
COMMENTS_EVERY = (6 * 60, 10 * 60)
COMMENT_POSTS = 8  # newest published posts whose comments are read
COMMENT_DAYS = 14
COMMENT_REPLIES_PER_DAY = 40
COMMENT_REPLIES_PER_ROUND = 5
DEFAULT_COMMENT_REPLY = "جواب‌تون رو دایرکت دادیم 🌹"
SENT_KEEP = 300


class _Ctx:
    """Enough of an OpContext for instagram_private outside an operation (no step log)."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.op = SimpleNamespace(payload={}, result={}, current_step="")

    @asynccontextmanager
    async def step(self, name: str, label_fa: str, request: Any = None):
        yield SimpleNamespace(response=None, message_fa="", request=None)


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt and not dt.tzinfo else dt


def _parse(ts: str | None) -> datetime | None:
    try:
        return _aware(datetime.fromisoformat(ts)) if ts else None
    except ValueError:
        return None


# ---------- settings ----------

async def bot_for(db: AsyncSession, channel: Channel) -> ChatBot | None:
    return (await db.execute(select(ChatBot).where(ChatBot.platform == "instagram", ChatBot.channel_id == channel.id)
                             .limit(1))).scalar_one_or_none()


async def configure(db: AsyncSession, channel: Channel, *, dm: bool, comments: bool, comment_text: str = "") -> ChatBot | None:
    """Turn DM / comment answering on or off for one Instagram account. Only messages after turning on count."""
    cfg = dict(channel.config or {})
    now = utcnow().isoformat()
    if dm and not cfg.get("dm_reply"):
        cfg["dm_since"] = now
    if comments and not cfg.get("comment_reply"):
        cfg["comments_since"] = now
    cfg.update(dm_reply=dm, comment_reply=comments, comment_text=(comment_text or "").strip()[:200])
    channel.config = cfg
    bot = await bot_for(db, channel)
    if dm or comments:
        if not bot:
            bot = ChatBot(store_id=channel.store_id, platform="instagram", mode="bot", channel_id=channel.id)
            db.add(bot)
        bot.status, bot.last_error = "connected", ""
        bot.bot_user_id = str(cfg.get("ig_user_id", ""))
        bot.bot_username = cfg.get("username", "")
        bot.bot_name = channel.display_name
    elif bot:
        bot.status = "disconnected"
    await db.flush()
    return bot


# ---------- incoming ----------

def _sent_by_us(channel: Channel, msg_id: str) -> bool:
    return msg_id in set((channel.config or {}).get("sent_ids") or [])


def remember_sent(channel: Channel, ids: list[str]) -> None:
    cfg = dict(channel.config or {})
    cfg["sent_ids"] = ([*(cfg.get("sent_ids") or []), *[i for i in ids if i]])[-SENT_KEEP:]
    channel.config = cfg


async def _product_for_media(db: AsyncSession, channel: Channel, media_pk: str) -> Product | None:
    if not media_pk:
        return None
    listing = (await db.execute(select(ChannelListing).where(
        ChannelListing.channel_id == channel.id, ChannelListing.remote_id == str(media_pk)).limit(1))).scalar_one_or_none()
    return await db.get(Product, listing.product_id) if listing else None


def _about(product: Product) -> str:
    return f"«{product.title[:80]}»" + (f" (کد {product.sku})" if product.sku else "")


async def inbound_for(db: AsyncSession, channel: Channel, thread, msg, me: str):
    """A DirectMessage as the chat runtime's Inbound; None for messages that are not for the assistant."""
    from app.services.chat_messenger import Inbound

    customer = next((u for u in thread.users if str(u.pk) != me), None)
    if not customer:
        return None
    text = (msg.text or "").strip()
    photo = ""
    if msg.media and msg.media.media_type == 1 and msg.media.thumbnail_url:
        photo = str(msg.media.thumbnail_url)
    shared = msg.media_share or msg.clip
    if shared is not None:
        product = await _product_for_media(db, channel, str(getattr(shared, "pk", "") or ""))
        if product:
            text = (text + f"\n[این پست را فرستاد: محصول {_about(product)}]").strip()
        else:
            line = next((x.strip() for x in (getattr(shared, "caption_text", "") or "").splitlines() if x.strip()), "")
            text = (text + f"\n[یک پست اینستاگرام فرستاد{': ' + line[:160] if line else ''}]").strip()
    if msg.reel_share:
        text = (text or str(msg.reel_share.get("text") or "")).strip()
        text = (text + "\n[در جواب استوری فروشگاه]").strip()
    unsupported = "" if text or photo else (msg.item_type or "other")
    return Inbound(chat_id=str(thread.id), message_id=str(msg.id), text=text, photo_file_id=photo,
                   from_id=str(msg.user_id or ""), name=customer.full_name or "", username=customer.username or "",
                   from_owner=str(msg.user_id) == me, unsupported=unsupported, raw={"item_type": msg.item_type})


async def _seen(db: AsyncSession, bot: ChatBot, thread_id: str, msg_id: str) -> bool:
    return bool(await db.scalar(select(ChatMessage.id).join(ChatConversation, ChatConversation.id == ChatMessage.conversation_id)
                                .where(ChatConversation.bot_id == bot.id, ChatConversation.external_chat_id == thread_id,
                                       ChatMessage.external_id == msg_id).limit(1)))


async def download(url: str, channel_id: int | None = None) -> tuple[bytes, str]:
    """A photo a customer sent (Instagram's CDN is filtered in Iran too, so through the proxies). An account with
    the seller's own proxy downloads through that one only."""
    own = next((n for n in proxy_pool.pool.nodes.values() if channel_id is not None and n.channel_id == channel_id), None)
    if own:
        if not own.url or own.error:
            raise PlatformError("پروکسی اختصاصی این حساب کار نمی‌کند؛ عکس دایرکت دانلود نشد.", "own_proxy_down")
        try:
            async with httpx.AsyncClient(timeout=60, proxy=own.url, follow_redirects=True) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise PlatformError("دانلود عکس دایرکت ناموفق بود.", "ig_photo") from exc
    else:
        resp = await proxy_pool.request("instagram", "GET", url, timeout=60, follow_redirects=True)
    if resp.status_code >= 400 or not resp.content:
        raise PlatformError("دانلود عکس دایرکت ناموفق بود.", "ig_photo")
    return resp.content, "photo.jpg"


async def _dms(ctx, db: AsyncSession, bot: ChatBot, channel: Channel, cl, creds) -> list[int]:
    from app.services.chat_runtime import receive

    since = _parse((channel.config or {}).get("dm_since")) or utcnow()
    me = str(cl.user_id)
    threads = await private._call(ctx, channel, cl, creds, cl.direct_threads, 20, thread_message_limit=10)
    pending = await private._call(ctx, channel, cl, creds, cl.direct_pending_inbox, 10)
    conv_ids: list[int] = []
    pending_ids = {str(t.id) for t in pending}
    for thread in [*threads, *pending]:
        if thread.is_group or (_aware(thread.last_activity_at) or since) <= since:
            continue
        fresh = sorted((m for m in thread.messages if (_aware(m.timestamp) or since) > since), key=lambda m: m.timestamp)
        if not fresh:
            continue
        if str(thread.id) in pending_ids:
            await private._call(ctx, channel, cl, creds, cl.direct_pending_approve, int(thread.id))
        for m in fresh:
            if _sent_by_us(channel, str(m.id)) or await _seen(db, bot, str(thread.id), str(m.id)):
                continue
            inbound = await inbound_for(db, channel, thread, m, me)
            if inbound and (cid := await receive(db, bot, inbound)) and cid not in conv_ids:
                conv_ids.append(cid)
    return conv_ids


# ---------- comments ----------

async def _comments(ctx, db: AsyncSession, bot: ChatBot, channel: Channel, cl, creds) -> list[int]:
    from app.services.chat_runtime import add_message

    cfg = dict(channel.config or {})
    since = _parse(cfg.get("comments_since")) or utcnow()
    me = str(cl.user_id)
    day = (utcnow() + timedelta(hours=3, minutes=30)).date().isoformat()
    done = int(cfg.get("cr_count") or 0) if cfg.get("cr_day") == day else 0
    handled = list(cfg.get("comment_ids") or [])
    reply_text = cfg.get("comment_text") or DEFAULT_COMMENT_REPLY
    listings = (await db.execute(select(ChannelListing).where(
        ChannelListing.channel_id == channel.id, ChannelListing.status.in_(("published", "out_of_sync")),
        ChannelListing.remote_id != "", ChannelListing.published_at >= utcnow() - timedelta(days=COMMENT_DAYS))
        .order_by(ChannelListing.published_at.desc()).limit(COMMENT_POSTS))).scalars().all()
    conv_ids: list[int] = []
    this_round = 0
    for listing in listings:
        if done >= COMMENT_REPLIES_PER_DAY or this_round >= COMMENT_REPLIES_PER_ROUND:
            break
        product = await db.get(Product, listing.product_id)
        comments = await private._call(ctx, channel, cl, creds, cl.media_comments, listing.remote_id, 20)
        for c in sorted(comments, key=lambda c: c.created_at_utc):
            if (str(c.user.pk) == me or c.replied_to_comment_id or str(c.pk) in handled
                    or (_aware(c.created_at_utc) or since) <= since):
                continue
            if done >= COMMENT_REPLIES_PER_DAY or this_round >= COMMENT_REPLIES_PER_ROUND:
                break
            handled.append(str(c.pk))
            if this_round:
                await asyncio.sleep(random.uniform(8, 25))  # not a burst of replies
            await private._call(ctx, channel, cl, creds, cl.media_comment, listing.remote_id,
                                f"@{c.user.username} {reply_text}", replied_to_comment_id=int(c.pk))
            done += 1
            this_round += 1
            conv = await _conv_for_user(db, bot, str(c.user.pk), c.user.username or "", c.user.full_name or "")
            about = f" محصول {_about(product)}" if product else ""
            add_message(db, conv, role="customer", text=f"{c.text.strip()}\n[کامنت زیر پست{about}]", external_id=f"c:{c.pk}")
            conv.seller_unread = True
            if conv.id not in conv_ids:
                conv_ids.append(conv.id)
    cfg = dict(channel.config or {})
    cfg.update(cr_day=day, cr_count=done, comment_ids=handled[-SENT_KEEP:], comments_checked_at=utcnow().isoformat())
    channel.config = cfg
    return conv_ids


async def _conv_for_user(db: AsyncSession, bot: ChatBot, user_pk: str, username: str, name: str) -> ChatConversation:
    """The commenter's DM conversation: the one we already have with them, or a new one addressed by user id
    ("u:<pk>") until the first reply creates the thread."""
    conv = None
    if username:
        conv = (await db.execute(select(ChatConversation).where(
            ChatConversation.bot_id == bot.id, ChatConversation.customer_username == username)
            .order_by(ChatConversation.id.desc()).limit(1))).scalar_one_or_none()
    if not conv:
        conv = (await db.execute(select(ChatConversation).where(
            ChatConversation.bot_id == bot.id, ChatConversation.external_chat_id == f"u:{user_pk}"))).scalar_one_or_none()
    if not conv:
        conv = ChatConversation(store_id=bot.store_id, bot_id=bot.id, platform="instagram", mode="bot",
                                external_chat_id=f"u:{user_pk}", customer_name=name, customer_username=username, state={})
        db.add(conv)
        await db.flush()
    return conv


# ---------- outgoing ----------

def _with_choices(text: str, buttons: list[list[str]] | None) -> str:
    """Instagram has no reply keyboard: the choices are written under the message."""
    labels = [b for row in (buttons or []) for b in row]
    return (text + ("\n\n" + "\n".join(f"• {b}" for b in labels) if labels else "")).strip()


async def _photo_file(url: str, folder: Path) -> Path:
    from app.services.images import load_bytes, to_jpeg

    p = folder / f"{random.randrange(10**9)}.jpg"
    p.write_bytes(to_jpeg(await load_bytes(url), max_side=1440))
    return p


async def deliver(db: AsyncSession, bot: ChatBot, conv: ChatConversation, replies, saved) -> None:
    """Sends the assistant's replies as DMs (raises PlatformError so the caller can record it)."""
    channel = await db.get(Channel, bot.channel_id)
    if not channel or channel.status != "connected":
        raise PlatformError("حساب اینستاگرام این گفتگو وصل نیست.", "bot_disconnected")
    ctx = _Ctx(db)
    cl, creds = await private._session(ctx, channel)
    sent: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="igdm-") as tmp:
            for n, (reply, row) in enumerate(zip(replies, saved)):
                if n:
                    await asyncio.sleep(random.uniform(1.5, 4))
                target = conv.external_chat_id
                kw = {"user_ids": [int(target[2:])]} if target.startswith("u:") else {"thread_ids": [int(target)]}
                text = _with_choices(reply.text or "", getattr(reply, "buttons", None))
                msg = None
                if reply.image_url:
                    try:
                        path = await _photo_file(reply.image_url, Path(tmp))
                        msg = await private._call(ctx, channel, cl, creds, cl.direct_send_photo, path, **kw)
                        sent.append(str(msg.id))
                    except PlatformError as exc:
                        if exc.code in ("ig_feedback", "ig_wait", "ig_blocked", "ig_challenge", "expired"):
                            raise
                if text:
                    msg = await private._call(ctx, channel, cl, creds, cl.direct_send, text, **kw)
                    sent.append(str(msg.id))
                if msg is not None:
                    row.external_id = str(msg.id)
                    if target.startswith("u:") and msg.thread_id:
                        await _adopt_thread(db, bot, conv, str(msg.thread_id))
    finally:
        remember_sent(channel, sent)
        private._save(channel, cl, creds)


async def _adopt_thread(db: AsyncSession, bot: ChatBot, conv: ChatConversation, thread_id: str) -> None:
    taken = await db.scalar(select(ChatConversation.id).where(
        ChatConversation.bot_id == bot.id, ChatConversation.external_chat_id == thread_id,
        ChatConversation.business_connection_id == conv.business_connection_id))
    if not taken:
        conv.external_chat_id = thread_id


async def send_text(db: AsyncSession, bot: ChatBot, conv: ChatConversation, text: str) -> str:
    """The seller answers from the panel."""
    row = SimpleNamespace(external_id="")
    await deliver(db, bot, conv, [SimpleNamespace(text=text, image_url="", buttons=None)], [row])
    return row.external_id


# ---------- polling ----------

async def poll_once(bot_id: int, *, comments_due: bool) -> list[int]:
    """One read of the inbox (and comments when due) for one account; returns conversations to answer."""
    async with SessionLocal() as db:
        bot = await db.get(ChatBot, bot_id)
        if not bot or bot.status != "connected" or not bot.channel_id:
            return []
        channel = await db.get(Channel, bot.channel_id)
        cfg = (channel.config or {}) if channel else {}
        if not channel or channel.status != "connected" or cfg.get("mode") != "private":
            return []
        paused = _parse(cfg.get("paused_until"))
        if paused and paused > utcnow():
            return []
        ctx = _Ctx(db)
        conv_ids: list[int] = []
        try:
            cl, creds = await private._session(ctx, channel)
            if cfg.get("dm_reply"):
                conv_ids += await _dms(ctx, db, bot, channel, cl, creds)
            if cfg.get("comment_reply") and comments_due:
                conv_ids += [c for c in await _comments(ctx, db, bot, channel, cl, creds) if c not in conv_ids]
            private._save(channel, cl, creds)
            bot.last_update_at, bot.last_error = utcnow(), ""
        except PlatformError as exc:
            bot.last_error = exc.message_fa
            log.info("instagram inbox %s: %s", bot_id, exc.message_fa)
        bot.last_check_at = utcnow()
        await db.commit()
        return conv_ids


async def _run(bot_id: int, stop: asyncio.Event) -> None:
    from app.services.chat_runtime import _sleep, schedule

    next_comments = 0.0
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        await _sleep(stop, random.uniform(*DM_EVERY))
        if stop.is_set():
            return
        due = loop.time() >= next_comments
        if due:
            next_comments = loop.time() + random.uniform(*COMMENTS_EVERY)
        try:
            for cid in await poll_once(bot_id, comments_due=due):
                schedule(cid)
        except Exception:  # noqa: BLE001 - keep reading; the next round may work
            log.exception("instagram inbox round failed for bot %s", bot_id)
        async with SessionLocal() as db:
            bot = await db.get(ChatBot, bot_id)
            if not bot or bot.status != "connected":
                return


async def poller_loop(stop: asyncio.Event) -> None:
    """One slow reader per Instagram account with answering on; picks up changes every 15 seconds."""
    from app.services.chat_runtime import _sleep

    tasks: dict[int, asyncio.Task] = {}
    while not stop.is_set():
        try:
            async with SessionLocal() as db:
                ids = set((await db.execute(select(ChatBot.id).where(
                    ChatBot.platform == "instagram", ChatBot.status == "connected"))).scalars())
            for bid in ids - tasks.keys():
                tasks[bid] = asyncio.create_task(_run(bid, stop))
            for bid, t in list(tasks.items()):
                if t.done():
                    tasks.pop(bid)
        except Exception:  # noqa: BLE001
            log.exception("instagram poller loop error")
        await _sleep(stop, 15)
    for t in tasks.values():
        t.cancel()
