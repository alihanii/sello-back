"""LinkedIn messages and comments answered by the chat assistant.

The seller turns it on per LinkedIn channel (messages, comments, or both). A ChatBot row (platform "linkedin",
channel_id = the channel) ties the account to the chat assistant, so its conversations show in the chat pages like
Telegram's and Instagram's. LinkedIn sends no webhooks, so the account is read on a slow, jittered timer through its
own session and fixed proxy (platforms/linkedin), never fast enough to look like a bot:
- messages: the inbox every 2.5 to 5 minutes; the assistant answers in the same conversation;
- comments: the account's recent published posts every 10 to 15 minutes; the assistant answers each new comment
  as a reply under it (LinkedIn does not let you message someone who is not a connection, so the answer stays public).
Both are one conversation per person, so a commenter who later writes a message keeps the same history.
A message the seller types on LinkedIn pauses the assistant for that chat."""

import asyncio
import hashlib
import logging
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal, utcnow
from app.models import Channel, ChannelListing, ChatBot, ChatConversation, ChatMessage, Product
from app.platforms.base import PlatformError
from app.platforms.linkedin import linkedin
from app.services.integrations import linkedin_app

log = logging.getLogger("sello.li_chat")

DM_EVERY = (150, 300)  # seconds
COMMENTS_EVERY = (10 * 60, 15 * 60)
COMMENT_POSTS = 6  # newest published posts whose comments are read
COMMENT_DAYS = 14
COMMENT_REPLIES_PER_DAY = 30
COMMENT_REPLIES_PER_ROUND = 5
CONVERSATIONS_PER_ROUND = 20
SENT_KEEP = 300
STOP_CODES = ("li_challenge", "li_wait", "li_restricted", "expired")


class _Ctx:
    """Enough of an OpContext for the adapter outside an operation (no step log)."""

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


def short_id(value: str) -> str:
    """LinkedIn ids are long URNs; messages keep a 40-character fingerprint (the column holds 60)."""
    return "li:" + hashlib.sha1(value.encode()).hexdigest()[:36] if value else ""


def _text_key(text: str) -> str:
    return "t:" + hashlib.sha1(" ".join((text or "").split()).encode()).hexdigest()[:20]


# ---------- settings ----------

async def bot_for(db: AsyncSession, channel: Channel) -> ChatBot | None:
    return (await db.execute(select(ChatBot).where(ChatBot.platform == "linkedin", ChatBot.channel_id == channel.id)
                             .limit(1))).scalar_one_or_none()


async def configure(db: AsyncSession, channel: Channel, *, dm: bool, comments: bool, comment_text: str = "") -> ChatBot | None:
    """Turn message / comment answering on or off for one LinkedIn account. Only what arrives after turning on counts.
    (`comment_text` is unused: on LinkedIn the assistant's whole answer goes under the comment.)"""
    cfg = dict(channel.config or {})
    now = utcnow().isoformat()
    if dm and not cfg.get("dm_reply"):
        cfg["dm_since"] = now
    if comments and not cfg.get("comment_reply"):
        cfg["comments_since"] = now
    cfg.update(dm_reply=dm, comment_reply=comments)
    channel.config = cfg
    bot = await bot_for(db, channel)
    if dm or comments:
        if not bot:
            bot = ChatBot(store_id=channel.store_id, platform="linkedin", mode="bot", channel_id=channel.id)
            db.add(bot)
        bot.status, bot.last_error = "connected", ""
        bot.bot_user_id = str(cfg.get("profile_id", ""))[:40]
        bot.bot_username = cfg.get("public_id", "")
        bot.bot_name = channel.display_name
    elif bot:
        bot.status = "disconnected"
    await db.flush()
    return bot


def remember_sent(channel: Channel, keys: list[str]) -> None:
    cfg = dict(channel.config or {})
    cfg["sent_ids"] = ([*(cfg.get("sent_ids") or []), *[k for k in keys if k]])[-SENT_KEEP:]
    channel.config = cfg


def _sent_by_us(channel: Channel, msg: dict[str, Any]) -> bool:
    sent = set((channel.config or {}).get("sent_ids") or [])
    return short_id(msg.get("id", "")) in sent or _text_key(msg.get("text", "")) in sent


# ---------- conversations ----------

async def conv_for(db: AsyncSession, bot: ChatBot, person: dict[str, Any]) -> ChatConversation:
    """One conversation per LinkedIn member, addressed by their profile id."""
    conv = (await db.execute(select(ChatConversation).where(
        ChatConversation.bot_id == bot.id, ChatConversation.external_chat_id == person["id"][:40]))).scalar_one_or_none()
    if not conv:
        conv = ChatConversation(store_id=bot.store_id, bot_id=bot.id, platform="linkedin", mode="bot",
                                external_chat_id=person["id"][:40], customer_name=person.get("name") or "",
                                customer_username=(person.get("public_id") or "")[:120], state={})
        db.add(conv)
        await db.flush()
    return conv


async def _seen(db: AsyncSession, conv: ChatConversation, ext: str) -> bool:
    return bool(await db.scalar(select(ChatMessage.id).where(
        ChatMessage.conversation_id == conv.id, ChatMessage.external_id == ext).limit(1)))


async def _dms(ctx, db: AsyncSession, bot: ChatBot, channel: Channel, session, creds, app) -> list[int]:
    from app.services.chat_runtime import add_message, pause_ai

    cfg = channel.config or {}
    me = cfg.get("profile_id", "")
    since = (_parse(cfg.get("dm_since")) or utcnow()).timestamp()
    convs = await linkedin.call(ctx, channel, session, creds, session.conversations, me, app["query_ids"])
    out: list[int] = []
    for c in convs[:CONVERSATIONS_PER_ROUND]:
        if c["last_activity"] and c["last_activity"] <= since:
            continue
        msgs = c["messages"]
        if not msgs or len(msgs) == 1 and c["last_activity"] > msgs[0]["at"] + 1:
            await asyncio.sleep(random.uniform(1.5, 4))
            msgs = await linkedin.call(ctx, channel, session, creds, session.messages, c["urn"], app["query_ids"])
        fresh = sorted((m for m in msgs if m["at"] > since), key=lambda m: m["at"])
        if not fresh:
            continue
        conv = await conv_for(db, bot, c["other"])
        conv.state = {**(conv.state or {}), "li_conversation": c["urn"]}
        answer = False
        for m in fresh:
            ext = short_id(m["id"])
            if _sent_by_us(channel, m) or await _seen(db, conv, ext):
                continue
            mine = m["sender_id"] == me
            text = m["text"].strip() or ("[پیوست]" if m["has_media"] else "")
            if not text:
                continue
            add_message(db, conv, role="seller" if mine else "customer", text=text, external_id=ext)
            if mine:
                await pause_ai(db, conv)  # the seller answered on LinkedIn themselves
                answer = False
            else:
                conv.state = {**(conv.state or {}), "li_via": "dm"}
                conv.seller_unread = True
                answer = True
        if answer and conv.id not in out:
            out.append(conv.id)
    return out


# ---------- comments ----------

async def _comments(ctx, db: AsyncSession, bot: ChatBot, channel: Channel, session, creds) -> list[int]:
    from app.services.chat_runtime import add_message

    cfg = dict(channel.config or {})
    me = cfg.get("profile_id", "")
    since = (_parse(cfg.get("comments_since")) or utcnow()).timestamp()
    day = (utcnow() + timedelta(hours=3, minutes=30)).date().isoformat()
    done = int(cfg.get("cr_count") or 0) if cfg.get("cr_day") == day else 0
    handled = list(cfg.get("comment_ids") or [])
    listings = (await db.execute(select(ChannelListing).where(
        ChannelListing.channel_id == channel.id, ChannelListing.status.in_(("published", "out_of_sync")),
        ChannelListing.remote_id != "", ChannelListing.published_at >= utcnow() - timedelta(days=COMMENT_DAYS))
        .order_by(ChannelListing.published_at.desc()).limit(COMMENT_POSTS))).scalars().all()
    out: list[int] = []
    this_round = 0
    for n, listing in enumerate(listings):
        if done >= COMMENT_REPLIES_PER_DAY or this_round >= COMMENT_REPLIES_PER_ROUND:
            break
        if n:
            await asyncio.sleep(random.uniform(2, 6))
        product = await db.get(Product, listing.product_id)
        for c in sorted(await linkedin.call(ctx, channel, session, creds, session.comments, listing.remote_id),
                        key=lambda c: c["created_at"]):
            if (c["author_id"] == me or c["is_reply"] or c["id"] in handled or not c["author_id"]
                    or c["created_at"] <= since or not c["text"].strip()):
                continue
            if done >= COMMENT_REPLIES_PER_DAY or this_round >= COMMENT_REPLIES_PER_ROUND:
                break
            handled.append(c["id"])
            done += 1
            this_round += 1
            conv = await conv_for(db, bot, {"id": c["author_id"], "name": c["author_name"], "public_id": c["author_public_id"]})
            about = f" محصول «{product.title[:80]}»" if product else ""
            add_message(db, conv, role="customer", text=f"{c['text'].strip()}\n[کامنت زیر پست لینکدین{about}]",
                        external_id=short_id("c:" + c["id"]))
            conv.state = {**(conv.state or {}), "li_via": "comment",
                          "li_comment": {"post": listing.remote_id, "id": c["id"], "name": c["author_name"]}}
            conv.seller_unread = True
            if conv.id not in out:
                out.append(conv.id)
    cfg = dict(channel.config or {})
    cfg.update(cr_day=day, cr_count=done, comment_ids=handled[-SENT_KEEP:], comments_checked_at=utcnow().isoformat())
    channel.config = cfg
    return out


# ---------- outgoing ----------

def _with_choices(text: str, buttons: list[list[str]] | None) -> str:
    """LinkedIn has no reply buttons: the choices are written under the message."""
    labels = [b for row in (buttons or []) for b in row]
    return (text + ("\n\n" + "\n".join(f"• {b}" for b in labels) if labels else "")).strip()


async def deliver(db: AsyncSession, bot: ChatBot, conv: ChatConversation, replies, saved) -> None:
    """Sends the assistant's replies: under the comment when the person last wrote a comment, else as a message.
    Photos cannot be attached here, so a reply with only a photo is sent as its link."""
    from app.services.media import public_url

    channel = await db.get(Channel, bot.channel_id)
    if not channel or channel.status != "connected":
        raise PlatformError("حساب لینکدین این گفتگو وصل نیست.", "bot_disconnected")
    ctx = _Ctx(db)
    session, creds = await linkedin.session(ctx, channel)
    state = dict(conv.state or {})
    via_comment = state.get("li_via") == "comment" and state.get("li_comment")
    texts = []
    for reply in replies:
        text = _with_choices(reply.text or "", getattr(reply, "buttons", None))
        if reply.image_url and not text:
            text = public_url(reply.image_url)
        texts.append(text)
    sent: list[str] = []
    me = (channel.config or {}).get("profile_id", "")
    try:
        if via_comment:
            # One public answer under the comment, not a stack of replies.
            body = "\n\n".join(t for t in texts if t)[:1200]
            first = (via_comment.get("name") or "").split(" ")[0]
            if body:
                await linkedin.call(ctx, channel, session, creds, session.reply_comment, via_comment["post"],
                                    via_comment["id"], (f"{first} عزیز، " if first else "") + body)
                for row in saved:
                    row.external_id = short_id("r:" + via_comment["id"])
            state.pop("li_comment", None)
            conv.state = state
        else:
            for n, (text, row) in enumerate(zip(texts, saved)):
                if not text:
                    continue
                if n:
                    await asyncio.sleep(random.uniform(1.5, 4))
                res = await linkedin.call(ctx, channel, session, creds, session.send_message, me, text,
                                          conversation_urn=state.get("li_conversation", ""),
                                          recipient_id="" if state.get("li_conversation") else conv.external_chat_id)
                row.external_id = short_id(res["id"]) if res["id"] else ""
                sent += [short_id(res["id"]), _text_key(text)]
                if res["conversation"] and not state.get("li_conversation"):
                    state["li_conversation"] = res["conversation"]
                    conv.state = dict(state)
    finally:
        remember_sent(channel, sent)
        linkedin.save(channel, session, creds)


async def send_text(db: AsyncSession, bot: ChatBot, conv: ChatConversation, text: str) -> str:
    """The seller answers from the panel (as a message, even after a comment)."""
    conv.state = {**(conv.state or {}), "li_via": "dm"}
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
        if not channel or channel.status != "connected" or channel.platform != "linkedin":
            return []
        paused = _parse(cfg.get("paused_until"))
        if paused and paused > utcnow():
            return []
        app = await linkedin_app(db)
        if not app["enabled"]:
            return []
        ctx = _Ctx(db)
        conv_ids: list[int] = []
        try:
            session, creds = await linkedin.session(ctx, channel)
            try:
                if cfg.get("dm_reply"):
                    conv_ids += await _dms(ctx, db, bot, channel, session, creds, app)
                if cfg.get("comment_reply") and comments_due:
                    conv_ids += [c for c in await _comments(ctx, db, bot, channel, session, creds) if c not in conv_ids]
            finally:
                linkedin.save(channel, session, creds)
            bot.last_update_at, bot.last_error = utcnow(), ""
        except PlatformError as exc:
            bot.last_error = exc.message_fa
            if exc.code == "expired":
                channel.status, channel.last_error = "expired", exc.message_fa
            log.info("linkedin inbox %s: %s", bot_id, exc.message_fa)
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
            log.exception("linkedin inbox round failed for bot %s", bot_id)
        async with SessionLocal() as db:
            bot = await db.get(ChatBot, bot_id)
            if not bot or bot.status != "connected":
                return


async def poller_loop(stop: asyncio.Event) -> None:
    """One slow reader per LinkedIn account with answering on; picks up changes every 15 seconds."""
    from app.services.chat_runtime import _sleep

    tasks: dict[int, asyncio.Task] = {}
    while not stop.is_set():
        try:
            async with SessionLocal() as db:
                ids = set((await db.execute(select(ChatBot.id).where(
                    ChatBot.platform == "linkedin", ChatBot.status == "connected"))).scalars())
            for bid in ids - tasks.keys():
                tasks[bid] = asyncio.create_task(_run(bid, stop))
            for bid, t in list(tasks.items()):
                if t.done():
                    tasks.pop(bid)
        except Exception:  # noqa: BLE001
            log.exception("linkedin poller loop error")
        await _sleep(stop, 15)
    for t in tasks.values():
        t.cancel()
