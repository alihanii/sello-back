"""Chat on social channels: which connected channels the chat assistant can answer in, and the on/off switches.

Telegram and Bale chat use their own bots (ChatBot rows made on the chat page). Social networks answer through the
channel the seller already connected for publishing (Instagram, LinkedIn); each platform registers a ChatSupport here and
the chat page lists those channels with one switch per feature (DMs, comments). Adding a platform = one register()
call with its own configure / state functions; the API and the page stay the same."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Channel, ChatBot


@dataclass
class Feature:
    key: str  # "dm" | "comments"
    label: str
    hint: str = ""


@dataclass
class ChatSupport:
    platform: str
    features: list[Feature]
    # (channel) -> "" when chat can run on this channel, else why not (shown to the seller)
    unavailable: Callable[[Channel], str]
    # (channel) -> {"dm": bool, "comments": bool, "comment_text": str}
    state: Callable[[Channel], dict[str, Any]]
    # (db, channel, dm, comments, comment_text) -> ChatBot | None; flushes, does not commit
    configure: Callable[..., Awaitable[ChatBot | None]]
    default_comment_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ChatSupport] = {}


def register(support: ChatSupport) -> None:
    _REGISTRY[support.platform] = support


def support_for(platform: str) -> ChatSupport | None:
    _load()
    return _REGISTRY.get(platform)


def platforms() -> list[str]:
    _load()
    return list(_REGISTRY)


def _load() -> None:
    if "instagram" not in _REGISTRY:
        from app.services import ig_chat

        register(ChatSupport(
            platform="instagram",
            features=[Feature("dm", "جواب خودکار دایرکت", "به پیام‌ها و درخواست‌های پیام جواب می‌دهد."),
                      Feature("comments", "جواب خودکار کامنت", "زیر کامنت جواب کوتاه می‌نویسد و جواب کامل را دایرکت می‌دهد.")],
            unavailable=lambda ch: "" if (ch.config or {}).get("mode") == "private"
            else "جواب خودکار فقط برای حسابی است که با نام کاربری و رمز وصل شده.",
            state=lambda ch: {"dm": bool((ch.config or {}).get("dm_reply")),
                              "comments": bool((ch.config or {}).get("comment_reply")),
                              "comment_text": (ch.config or {}).get("comment_text", "")},
            configure=ig_chat.configure,
            default_comment_text=ig_chat.DEFAULT_COMMENT_REPLY,
        ))


    if "linkedin" not in _REGISTRY:
        from app.services import li_chat

        register(ChatSupport(
            platform="linkedin",
            features=[Feature("dm", "جواب خودکار پیام‌ها", "به پیام‌های خصوصی لینکدین جواب می‌دهد."),
                      Feature("comments", "جواب خودکار کامنت",
                              "زیر کامنت‌های پست‌ها جواب می‌دهد (لینکدین پیام به غیرمخاطب را اجازه نمی‌دهد).")],
            unavailable=lambda ch: "",
            state=lambda ch: {"dm": bool((ch.config or {}).get("dm_reply")),
                              "comments": bool((ch.config or {}).get("comment_reply")), "comment_text": ""},
            configure=li_chat.configure,
        ))


async def bot_for(db: AsyncSession, channel: Channel) -> ChatBot | None:
    return (await db.execute(select(ChatBot).where(ChatBot.platform == channel.platform, ChatBot.channel_id == channel.id)
                             .limit(1))).scalar_one_or_none()


async def channel_out(db: AsyncSession, channel: Channel) -> dict[str, Any]:
    support = support_for(channel.platform)
    assert support
    bot = await bot_for(db, channel)
    st = support.state(channel)
    why = support.unavailable(channel) if channel.status == "connected" else "کانال وصل نیست؛ از صفحهٔ کانال‌ها دوباره وصلش کن."
    active = bool(not why and (st.get("dm") or st.get("comments")))
    return {
        "channel_id": channel.id, "platform": channel.platform, "display_name": channel.display_name,
        "username": (channel.config or {}).get("username", ""), "channel_status": channel.status,
        "available": not why, "unavailable_reason": why,
        "features": [{"key": f.key, "label": f.label, "hint": f.hint} for f in support.features],
        "dm": st.get("dm", False), "comments": st.get("comments", False),
        "comment_text": st.get("comment_text", ""), "default_comment_text": support.default_comment_text,
        "active": active,
        "last_update_at": bot.last_update_at if bot and active else None,
        "last_error": (bot.last_error if bot and active else "") or "",
    }


async def list_for_store(db: AsyncSession, store_id: int) -> list[dict[str, Any]]:
    rows = (await db.execute(select(Channel).where(Channel.store_id == store_id, Channel.platform.in_(platforms()),
                                                   Channel.status != "disconnected").order_by(Channel.id))).scalars().all()
    return [await channel_out(db, ch) for ch in rows]
