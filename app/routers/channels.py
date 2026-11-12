from datetime import timedelta
from typing import Any

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import settings
from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.core.security import encrypt_credentials
from app.models import Channel, Operation, ProxyNode
from app.platforms.base import PlatformError
from app.platforms.digikala import digikala
from app.platforms.instagram import authorize_url
from app.platforms.instagram_private import submit_code
from app.platforms.registry import PLATFORM_INFO
from app.schemas import ChannelOut, OperationOut
from app.services import ig_chat, li_chat
from app.services.integrations import instagram_app, instagram_crawler, linkedin_app
from app.services.ops import OpContext, enqueue, fail_operation
from app.services.proxy_links import LinkError, parse_link
from app.services.proxy_pool import pool
from app.services.plans import check_limit

router = APIRouter(tags=["channels"])


@router.get("/platforms")
async def platforms(db: DB):
    """Instagram opens for publishing when the admin's Meta app is on, and for crawl only
    (publish off) when just the admin's crawl key is set."""
    ig = await instagram_app(db)
    crawl_ready = bool((await instagram_crawler(db))["token"])
    li = await linkedin_app(db)
    out = []
    for k, v in PLATFORM_INFO.items():
        row = {"platform": k, **v, "enabled": not v.get("flag") or ig["enabled"]}
        if k == "instagram":
            row["login"] = ig["method"] == "login"  # username + password (Instagram app API): posts, stories, reels
            row["oauth"] = ig["method"] == "oauth" and ig["oauth_ready"]
            row["formats"] = ["post", "story", "reel"] if row["login"] else ["post"]
            row["publish_ready"] = ig["enabled"]
            row["crawl_ready"] = crawl_ready
            row["crawl_only"] = crawl_ready and not ig["enabled"]
            row["enabled"] = ig["enabled"] or crawl_ready
        if k == "linkedin":
            row["enabled"] = row["publish_ready"] = li["enabled"]
            row["formats"] = ["post"]
        out.append(row)
    return out


@router.get("/stores/{store_id}/channels/instagram/oauth-url")
async def instagram_oauth_url(store_id: int, user: CurrentUser, db: DB, channel_id: int | None = None):
    await get_store_for(db, user, store_id)
    app = await instagram_app(db)
    if not (app["enabled"] and app["oauth_ready"]):
        raise HTTPException(400, "ورود با اینستاگرام هنوز توسط ادمین تنظیم نشده است.")
    state = jwt.encode({"sid": store_id, "uid": user.id, "cid": channel_id, "purpose": "ig_oauth",
                        "exp": utcnow() + timedelta(minutes=15)}, settings.jwt_secret, algorithm="HS256")
    return {"url": authorize_url(app, state)}


class OAuthIn(BaseModel):
    code: str
    state: str


@router.post("/channels/instagram/oauth-callback", response_model=OperationOut, status_code=202)
async def instagram_oauth_callback(body: OAuthIn, user: CurrentUser, db: DB):
    try:
        st = jwt.decode(body.state, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(400, "لینک ورود منقضی شده؛ دوباره از صفحهٔ کانال‌ها تلاش کنید.")
    if st.get("purpose") != "ig_oauth" or st.get("uid") != user.id:
        raise HTTPException(403, "این ورود متعلق به حساب شما نیست.")
    return await connect(st["sid"], ConnectIn(platform="instagram", channel_id=st.get("cid"), data={"code": body.code}),
                         user, db)


@router.get("/stores/{store_id}/channels", response_model=list[ChannelOut])
async def list_channels(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    q = select(Channel).where(Channel.store_id == store_id, Channel.status != "disconnected").order_by(Channel.id)
    return (await db.execute(q)).scalars().all()


class ConnectIn(BaseModel):
    platform: str
    channel_id: int | None = None  # reconnect an existing channel
    data: dict[str, Any] = {}


async def _channel(db, user, store_id: int, body: ConnectIn) -> Channel:
    if body.platform not in PLATFORM_INFO:
        raise HTTPException(422, "پلتفرم پشتیبانی نمی‌شود.")
    if body.channel_id:
        ch = await db.get(Channel, body.channel_id)
        if not ch or ch.store_id != store_id:
            raise HTTPException(404, "کانال پیدا نشد.")
        return ch
    await check_limit(db, user, "channels", store_id=store_id, platform=body.platform)
    ch = Channel(store_id=store_id, platform=body.platform, status="disconnected", config={})
    db.add(ch)
    await db.flush()
    return ch


@router.post("/stores/{store_id}/channels/digikala/send-code")
async def digikala_send_code(store_id: int, body: ConnectIn, user: CurrentUser, db: DB):
    """Step 1 of Digikala OTP; runs inline because the seller is waiting for the SMS."""
    await get_store_for(db, user, store_id)
    body.platform = "digikala"
    ch = await _channel(db, user, store_id, body)
    op = await enqueue(db, store_id=store_id, type="connect", channel_id=ch.id, title_fa="ارسال کد تأیید دیجی‌کالا")
    op.status = "running"
    await db.commit()
    ctx = OpContext(db, op)
    try:
        pending = await digikala.send_code(ctx, body.data.get("phone", ""))
    except PlatformError as exc:
        fail_operation(op, exc)
        await db.commit()
        raise HTTPException(400, {"message": exc.message_fa, "operation_id": op.id})
    ch.config = {**ch.config, "pending": pending}
    op.status = "succeeded"
    await db.commit()
    return {"channel_id": ch.id, "operation_id": op.id}


async def _own_proxy(db, ch: Channel, link: str) -> None:
    """The seller's own proxy (their fixed IP) for this Instagram account: checked against Instagram now, then the
    account uses only it. A v2ray link runs through the same Xray core as the admin's proxies."""
    try:
        parsed = parse_link(link)
    except LinkError as exc:
        raise HTTPException(422, f"کانفیگ پروکسی قابل استفاده نیست: {exc}")
    row = (await db.execute(select(ProxyNode).where(ProxyNode.channel_id == ch.id))).scalar_one_or_none()
    if not row:
        row = ProxyNode(channel_id=ch.id, purposes=["instagram"], enabled=True, health={}, link_encrypted="")
        db.add(row)
    row.name = f"اختصاصی کانال {ch.id}"
    row.protocol, row.address = parsed.protocol, parsed.address[:255]
    row.link_encrypted, row.health = encrypt_credentials({"link": link}), {}
    await db.flush()
    ch.config = {**(ch.config or {}), "own_proxy": True, "proxy_node_id": row.id, "proxy_address": parsed.address}
    await db.commit()
    await pool.reload()
    res = await pool.check(row.id, "instagram")
    await pool.persist()
    _proxy_health(ch, res)
    await db.commit()
    if not res.get("ok"):
        raise HTTPException(422, "با این کانفیگ به اینستاگرام وصل نشد" + (f": {res.get('error')}" if res.get("error") else "")
                            + ". کانفیگ را بررسی کن یا خالی بگذار تا از پروکسی‌های فروشگاه استفاده شود.")


async def _ig_channel(db, user, channel_id: int) -> Channel:
    ch = await db.get(Channel, channel_id)
    if not ch or ch.platform != "instagram":
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    return ch


def _proxy_health(ch: Channel, res: dict) -> None:
    ch.config = {**(ch.config or {}), "proxy_health": {k: res.get(k) for k in ("ok", "latency_ms", "error", "checked_at")}}


class ProxyIn(BaseModel):
    link: str = Field(min_length=8, max_length=4000)


@router.put("/channels/{channel_id}/instagram/proxy", response_model=ChannelOut)
async def set_own_proxy(channel_id: int, body: ProxyIn, user: CurrentUser, db: DB):
    """Set or change the seller's own proxy. The new config is checked first; if it fails, the old setting stays."""
    ch = await _ig_channel(db, user, channel_id)
    if (ch.config or {}).get("mode") != "private":
        raise HTTPException(422, "پروکسی اختصاصی فقط برای حسابی است که با نام کاربری و رمز وصل شده.")
    old_cfg = dict(ch.config or {})
    row = (await db.execute(select(ProxyNode).where(ProxyNode.channel_id == ch.id))).scalar_one_or_none()
    old_row = (row.protocol, row.address, row.link_encrypted, dict(row.health or {})) if row else None
    try:
        await _own_proxy(db, ch, body.link.strip())
    except HTTPException:
        row = (await db.execute(select(ProxyNode).where(ProxyNode.channel_id == ch.id))).scalar_one_or_none()
        if old_row and row:
            row.protocol, row.address, row.link_encrypted, row.health = old_row
        elif row:
            await db.delete(row)
        ch.config = old_cfg
        await db.commit()
        await pool.reload()
        raise
    ch.config = {k: v for k, v in ch.config.items() if k != "bad_nodes"}
    await db.commit()
    await db.refresh(ch)
    return ch


@router.post("/channels/{channel_id}/instagram/proxy/test")
async def test_own_proxy(channel_id: int, user: CurrentUser, db: DB):
    """Check the account's proxy against Instagram now: the seller's own one, or the shared one it is pinned to."""
    ch = await _ig_channel(db, user, channel_id)
    cfg = ch.config or {}
    node_id = cfg.get("proxy_node_id")
    if not node_id or node_id not in pool.nodes:
        if cfg.get("own_proxy"):
            res = {"ok": False, "latency_ms": None, "error": "پروکسی اختصاصی بارگذاری نشد؛ کانفیگ را دوباره ثبت کن.",
                   "checked_at": utcnow().isoformat()}
        else:
            raise HTTPException(422, "این حساب هنوز به پروکسی‌ای وصل نشده.")
    else:
        res = await pool.check(node_id, "instagram")
        await pool.persist()
    _proxy_health(ch, res)
    await db.commit()
    return {**res, "own": bool(cfg.get("own_proxy")), "address": cfg.get("proxy_address", "") if cfg.get("own_proxy") else ""}


@router.delete("/channels/{channel_id}/instagram/proxy", response_model=ChannelOut)
async def remove_own_proxy(channel_id: int, user: CurrentUser, db: DB):
    """Back to the admin's shared proxies."""
    ch = await db.get(Channel, channel_id)
    if not ch or ch.platform != "instagram":
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    for row in (await db.execute(select(ProxyNode).where(ProxyNode.channel_id == ch.id))).scalars():
        await db.delete(row)
    ch.config = {k: v for k, v in (ch.config or {}).items() if k not in ("own_proxy", "proxy_node_id", "proxy_address", "proxy_health")}
    await db.commit()
    await pool.reload()
    await db.refresh(ch)
    return ch


@router.post("/stores/{store_id}/channels/connect", response_model=OperationOut, status_code=202)
async def connect(store_id: int, body: ConnectIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    ch = await _channel(db, user, store_id, body)
    if body.platform == "instagram" and body.data.get("username"):
        link = (body.data.pop("proxy", "") or "").strip()
        if link:
            try:
                await _own_proxy(db, ch, link)
            except HTTPException:
                if not body.channel_id:  # a new channel that never connected: don't leave it behind
                    for row in (await db.execute(select(ProxyNode).where(ProxyNode.channel_id == ch.id))).scalars():
                        await db.delete(row)
                    await db.delete(ch)
                    await db.commit()
                    await pool.reload()
                raise
    label = PLATFORM_INFO[body.platform]["label"]
    op = await enqueue(db, store_id=store_id, type="connect", channel_id=ch.id, payload={"data": body.data},
                       title_fa=f"اتصال {label}")
    await db.commit()
    return op


@router.post("/channels/{channel_id}/health", response_model=OperationOut, status_code=202)
async def health(channel_id: int, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    op = await enqueue(db, store_id=ch.store_id, type="health_check", channel_id=ch.id,
                       title_fa=f"بررسی سلامت {ch.display_name or ch.platform}")
    await db.commit()
    return op


class CodeIn(BaseModel):
    code: str


@router.post("/channels/{channel_id}/instagram/code")
async def instagram_code(channel_id: int, body: CodeIn, user: CurrentUser, db: DB):
    """The code Instagram sent (2FA or a login check) while a username + password connect is waiting for it."""
    ch = await db.get(Channel, channel_id)
    if not ch or ch.platform != "instagram":
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    if not submit_code(ch.id, body.code):
        raise HTTPException(409, "ورود منتظر کد نیست یا زمانش تمام شده؛ دوباره وصل شو.")
    return {"ok": True}


class InboxIn(BaseModel):
    dm: bool = False
    comments: bool = False
    comment_text: str = ""


@router.put("/channels/{channel_id}/instagram/inbox", response_model=ChannelOut)
async def instagram_inbox(channel_id: int, body: InboxIn, user: CurrentUser, db: DB):
    """The chat assistant answers this account's DMs and/or comments (username + password accounts only)."""
    ch = await db.get(Channel, channel_id)
    if not ch or ch.platform != "instagram":
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    if (ch.config or {}).get("mode") != "private":
        raise HTTPException(422, "جواب خودکار فقط برای حسابی است که با نام کاربری و رمز وصل شده.")
    await ig_chat.configure(db, ch, dm=body.dm, comments=body.comments, comment_text=body.comment_text)
    await db.commit()
    await db.refresh(ch)
    return ch


class LinkedInInboxIn(BaseModel):
    dm: bool = False
    comments: bool = False


@router.put("/channels/{channel_id}/linkedin/inbox", response_model=ChannelOut)
async def linkedin_inbox(channel_id: int, body: LinkedInInboxIn, user: CurrentUser, db: DB):
    """The chat assistant answers this LinkedIn account's messages and/or the comments under its posts."""
    ch = await db.get(Channel, channel_id)
    if not ch or ch.platform != "linkedin":
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    await li_chat.configure(db, ch, dm=body.dm, comments=body.comments)
    await db.commit()
    await db.refresh(ch)
    return ch


class ChannelPatch(BaseModel):
    display_name: str | None = None


@router.patch("/channels/{channel_id}", response_model=ChannelOut)
async def update_channel(channel_id: int, body: ChannelPatch, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    if body.display_name is not None:
        ch.display_name = body.display_name
    await db.commit()
    await db.refresh(ch)
    return ch


@router.delete("/channels/{channel_id}")
async def disconnect(channel_id: int, user: CurrentUser, db: DB):
    ch = await db.get(Channel, channel_id)
    if not ch:
        raise HTTPException(404, "کانال پیدا نشد.")
    await get_store_for(db, user, ch.store_id)
    ch.status, ch.credentials_encrypted = "disconnected", None
    if ch.platform == "instagram" and (bot := await ig_chat.bot_for(db, ch)):
        bot.status = "disconnected"
    if ch.platform == "linkedin" and (bot := await li_chat.bot_for(db, ch)):
        bot.status = "disconnected"
    await db.commit()
    return {"ok": True}
