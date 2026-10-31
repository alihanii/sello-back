"""Telegram/Bale Bot API calls for the chat assistant, and turning raw updates into one simple shape.

Two ways in (spec §2): the store's own bot (`message` updates from private chats) and Telegram Business
(`business_connection` + `business_message`), where the bot answers in the seller's personal chats."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.security import decrypt_credentials
from app.models import ChatBot
from app.platforms.base import PlatformError
from app.platforms.botapi import BotApiAdapter, bale, telegram

ADAPTERS: dict[str, BotApiAdapter] = {"telegram": telegram, "bale": bale}
# Bale's Bot API has no business mode (docs.bale.ai lists only message, edited_message, callback_query).
BUSINESS_PLATFORMS = {"telegram"}
ALLOWED_UPDATES = ["message", "business_connection", "business_message"]
MAX_PHOTO_BYTES = 10 * 1024 * 1024


def token_hash(token: str) -> str:
    return hashlib.sha256(token.strip().encode()).hexdigest()


def bot_token(bot: ChatBot) -> str:
    token = decrypt_credentials(bot.credentials_encrypted).get("bot_token", "")
    if not token:
        raise PlatformError("توکن ربات ذخیره نشده است؛ دوباره وصل کن.", "missing_token")
    return token


async def call(bot: ChatBot | str, method: str, data: dict[str, Any] | None = None, *, token: str = "",
               files: dict[str, Any] | None = None) -> Any:
    """`bot` is a ChatBot or a platform name (when only a token is known, e.g. while connecting)."""
    platform = bot if isinstance(bot, str) else bot.platform
    return await ADAPTERS[platform]._call(token or bot_token(bot), method, data, files)  # noqa: SLF001


async def get_updates(bot: ChatBot, offset: int, timeout: int = 25) -> list[dict[str, Any]]:
    adapter = ADAPTERS[bot.platform]
    url = f"{adapter.base_url}/bot{bot_token(bot)}/getUpdates"
    body = {"offset": offset, "timeout": timeout, "limit": 50, "allowed_updates": ALLOWED_UPDATES}
    async with httpx.AsyncClient(timeout=timeout + 15) as client:
        resp = await client.post(url, json=body)
    try:
        data = resp.json()
    except ValueError:
        raise PlatformError(f"{adapter.label_fa} پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])
    if not data.get("ok"):
        raise adapter._humanize(data)  # noqa: SLF001
    return data.get("result") or []


async def send_text(bot: ChatBot, chat_id: str, text: str, business_connection_id: str = "") -> str:
    data: dict[str, Any] = {"chat_id": chat_id, "text": text[:4000]}
    if business_connection_id:
        data["business_connection_id"] = business_connection_id
    result = await call(bot, "sendMessage", data)
    return str((result or {}).get("message_id", ""))


async def send_photo(bot: ChatBot, chat_id: str, image: bytes, filename: str, caption: str = "",
                     business_connection_id: str = "") -> str:
    data: dict[str, Any] = {"chat_id": str(chat_id), "caption": caption[:1000]}
    if business_connection_id:
        data["business_connection_id"] = business_connection_id
    result = await call(bot, "sendPhoto", data, files={"photo": (filename, image)})
    return str((result or {}).get("message_id", ""))


async def download(bot: ChatBot, file_id: str) -> tuple[bytes, str]:
    adapter = ADAPTERS[bot.platform]
    token = bot_token(bot)
    info = await call(bot, "getFile", {"file_id": file_id}, token=token)
    if int(info.get("file_size") or 0) > MAX_PHOTO_BYTES:
        raise PlatformError("حجم عکس بیش از ۱۰ مگابایت است.", "image_too_large")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(f"{adapter.file_base_url}/file/bot{token}/{info['file_path']}")
        resp.raise_for_status()
    return resp.content, info["file_path"].rsplit("/", 1)[-1]


# ---------- updates ----------


@dataclass
class Inbound:
    chat_id: str
    message_id: str
    text: str = ""
    photo_file_id: str = ""
    from_id: str = ""
    name: str = ""
    username: str = ""
    business_connection_id: str = ""
    from_owner: bool = False  # Telegram Business: the seller typed it in their own chat
    unsupported: str = ""  # sticker | voice | video | document … (answered with a fixed text, no model)
    raw: dict[str, Any] = field(default_factory=dict)


def _name(user: dict[str, Any]) -> str:
    return " ".join(x for x in (user.get("first_name", ""), user.get("last_name", "")) if x).strip()


def parse_message(bot: ChatBot, msg: dict[str, Any], business: bool) -> Inbound | None:
    chat = msg.get("chat") or {}
    if chat.get("type") not in (None, "private"):
        return None  # groups and channels are not customer chats
    sender = msg.get("from") or {}
    photo = (msg.get("photo") or [{}])[-1].get("file_id", "")
    unsupported = ""
    if not (msg.get("text") or photo):
        unsupported = next((k for k in ("sticker", "voice", "video", "video_note", "audio", "document", "location",
                                        "contact", "animation") if msg.get(k)), "other")
    inbound = Inbound(
        chat_id=str(chat.get("id", "")), message_id=str(msg.get("message_id", "")),
        text=(msg.get("text") or msg.get("caption") or "").strip(), photo_file_id=photo,
        from_id=str(sender.get("id", "")), name=_name(chat) or _name(sender),
        username=chat.get("username") or sender.get("username") or "",
        business_connection_id=msg.get("business_connection_id", "") if business else "",
        unsupported=unsupported, raw=msg,
    )
    if business:
        inbound.from_owner = bool(bot.business_user_id) and inbound.from_id == bot.business_user_id
    if contact := msg.get("contact"):
        inbound.text = (inbound.text + " " + str(contact.get("phone_number", ""))).strip()
        inbound.unsupported = ""
    return inbound


def business_rights(conn: dict[str, Any]) -> bool:
    """Bot API 9 moved can_reply into `rights`; older servers send it at the top level."""
    rights = conn.get("rights") or {}
    return bool(rights.get("can_reply", conn.get("can_reply", False)))
