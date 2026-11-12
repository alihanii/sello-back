"""Instagram through its own app API (instagrapi): the seller signs in with username and password, any account type.

This is not Meta's public API, so everything here is built to look like one phone using the account normally:
- one login, then the saved session (cookies + the same device ids) is reused; a fresh login only when Instagram
  drops the session, and never in a loop;
- every account goes through one fixed proxy from the admin's pool (admin panel › proxies), so Instagram always sees
  the same IP; it moves to another node only when its node is down;
- publishes are paced: a random gap between two publishes of one account and a daily cap (admin panel settings);
- when Instagram pushes back (challenge, "feedback required", "please wait"), the account is paused and the seller
  is told, instead of retrying.

instagrapi is synchronous, so its calls run in a thread."""

import asyncio
import logging
import random
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, PlatformError, PublishResult
from app.services import proxy_pool, video
from app.services.images import load_bytes, story_frame, to_jpeg
from app.services.text import caption_for, to_en_digits

log = logging.getLogger("sello.instagram")

CAPTION_LIMIT = 2200
MAX_ALBUM = 10
CODE_WAIT_SECONDS = 300
PAUSE_HOURS = {"feedback": 24, "wait": 2, "challenge": 0, "sentry": 48}
PROXY_HELP = {"type": "help", "label": "پروکسی اینستاگرام در پنل ادمین"}
RETRY = {"type": "retry", "label": "تلاش دوباره"}
FORMATS = ("post", "story", "reel")
FORMAT_FA = {"post": "پست", "story": "استوری", "reel": "ریلز"}


# ---------- codes the seller types while connecting (2FA / challenge) ----------

class _CodeWait:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.code = ""


_waits: dict[int, _CodeWait] = {}


def submit_code(channel_id: int, code: str) -> bool:
    """Called by the API when the seller types the code Instagram sent; False when nothing is waiting."""
    w = _waits.get(channel_id)
    if not w:
        return False
    w.code = "".join(ch for ch in to_en_digits(code) if ch.isascii() and ch.isdigit())
    w.event.set()
    return True


def waiting_for_code(channel_id: int) -> bool:
    return channel_id in _waits


# ---------- pacing ----------

_locks: dict[int, asyncio.Lock] = {}


def _day(now: datetime) -> str:
    return (now + timedelta(hours=3, minutes=30)).date().isoformat()  # the seller's day (Tehran)


def _parse(ts: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def next_slot(channel: Channel, app: dict[str, Any], now: datetime) -> datetime | None:
    """When this account may publish next; None = now. Pure, so it can be tested."""
    cfg = channel.config or {}
    waits = [t for t in (_parse(cfg.get("paused_until")), _parse(cfg.get("next_at"))) if t and t > now]
    if cfg.get("day") == _day(now) and int(cfg.get("count") or 0) >= app["daily_limit"]:
        local_midnight = datetime.combine(datetime.fromisoformat(_day(now)) + timedelta(days=1), datetime.min.time())
        waits.append((local_midnight - timedelta(hours=3, minutes=30)).replace(tzinfo=now.tzinfo)
                     + timedelta(minutes=random.randint(30, 120)))
    return max(waits) if waits else None


def reserve(channel: Channel, app: dict[str, Any], now: datetime) -> None:
    """Take today's slot and push the next one a random gap away."""
    cfg = dict(channel.config or {})
    count = int(cfg.get("count") or 0) if cfg.get("day") == _day(now) else 0
    gap = random.randint(app["gap_min_minutes"], max(app["gap_min_minutes"], app["gap_max_minutes"]))
    cfg.update(day=_day(now), count=count + 1, next_at=(now + timedelta(minutes=gap)).isoformat())
    channel.config = cfg


def pause(channel: Channel, kind: str, reason: str) -> None:
    cfg = dict(channel.config or {})
    hours = PAUSE_HOURS.get(kind, 24)
    if hours:
        cfg["paused_until"] = (utcnow() + timedelta(hours=hours)).isoformat()
    cfg["pause_reason"] = reason
    channel.config = cfg


# ---------- errors ----------

def _humanize(exc: Exception, channel: Channel | None = None) -> PlatformError:
    import requests
    from instagrapi import exceptions as e

    name = type(exc).__name__
    if isinstance(exc, PlatformError):
        return exc
    if isinstance(exc, e.BadPassword):
        return PlatformError("نام کاربری یا رمز اینستاگرام اشتباه است.", "bad_password", RECONNECT)
    if isinstance(exc, e.UserNotFound):
        return PlatformError("این نام کاربری در اینستاگرام پیدا نشد.", "user_not_found", RECONNECT)
    if isinstance(exc, (e.FeedbackRequired,)):
        msg = "اینستاگرام موقتاً اجازهٔ انتشار نمی‌دهد (Feedback required). حساب ۲۴ ساعت استراحت می‌کند تا محدود نشود."
        if channel:
            pause(channel, "feedback", msg)
        return PlatformError(msg, "ig_feedback", RETRY, name)
    if isinstance(exc, (e.PleaseWaitFewMinutes, e.RateLimitError, e.ClientThrottledError)):
        msg = "اینستاگرام گفت کمی صبر کنید؛ انتشارهای این حساب دو ساعت عقب افتاد."
        if channel:
            pause(channel, "wait", msg)
        return PlatformError(msg, "ig_wait", RETRY, name)
    if isinstance(exc, e.SentryBlock):
        msg = "اینستاگرام این حساب را موقتاً بلاک کرد؛ ۴۸ ساعت چیزی منتشر نمی‌شود. در اپ اینستاگرام وارد شو و هشدارها را تأیید کن."
        if channel:
            pause(channel, "sentry", msg)
        return PlatformError(msg, "ig_blocked", RECONNECT, name)
    if isinstance(exc, (e.ChallengeRequired, e.ChallengeError, e.RecaptchaChallengeForm,
                        e.SelectContactPointRecoveryForm, e.ChallengeUnknownStep)):
        msg = ("اینستاگرام تأیید هویت خواست. اپ اینستاگرام را روی گوشی باز کن، اگر پیغام «این شما بودید؟» آمد "
               "تأیید کن، بعد دوباره وصل شو.")
        if channel:
            pause(channel, "challenge", msg)
        return PlatformError(msg, "ig_challenge", RECONNECT, name)
    if isinstance(exc, e.TwoFactorRequired):
        return PlatformError("کد تأیید دومرحله‌ای وارد نشد یا اشتباه بود.", "ig_2fa", RECONNECT, name)
    if isinstance(exc, e.LoginRequired):
        return PlatformError("نشست اینستاگرام تمام شد؛ دوباره وصل شو.", "expired", RECONNECT, name)
    if isinstance(exc, e.ProxyAddressIsBlocked):
        if channel and (channel.config or {}).get("own_proxy"):
            return PlatformError("اینستاگرام IP پروکسی اختصاصی این حساب را بسته؛ کانفیگ دیگری بده و دوباره وصل شو.",
                                 "ig_proxy_blocked", RECONNECT, name)
        if channel:
            cfg = dict(channel.config or {})
            cfg["bad_nodes"] = sorted(set(cfg.get("bad_nodes") or []) | {cfg.get("proxy_node_id")} - {None})
            cfg.pop("proxy_node_id", None)
            channel.config = cfg
        return PlatformError("اینستاگرام IP این پروکسی را بسته؛ دفعهٔ بعد پروکسی دیگری از پنل ادمین استفاده می‌شود.",
                             "ig_proxy_blocked", PROXY_HELP, name)
    if isinstance(exc, (e.ClientConnectionError, e.ClientRequestTimeout, requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout, OSError)) or \
            "ConnectionError" in name or "ProxyError" in name or "Timeout" in name:
        if channel and (channel.config or {}).get("own_proxy"):
            return PlatformError("ارتباط با اینستاگرام از پروکسی اختصاصی این حساب برقرار نشد؛ در صفحهٔ کانال‌ها «تست» یا «تغییر» پروکسی را بزن.",
                                 "network", RECONNECT, str(exc)[:300])
        return PlatformError("ارتباط با اینستاگرام از طریق پروکسی برقرار نشد؛ پروکسی‌های اینستاگرام را در پنل ادمین بررسی کن.",
                             "network", PROXY_HELP, str(exc)[:300])
    return PlatformError(f"اینستاگرام خطا داد: {str(exc)[:200] or name}", "platform_error", detail=name)


# ---------- the client ----------

def _socks(url: str) -> str:
    """Resolve names through the proxy too: Instagram's domains are poisoned in Iran's DNS."""
    return "socks5h://" + url.removeprefix("socks5://") if url.startswith("socks5://") else url


async def _route(ctx, channel: Channel) -> str:
    cfg = channel.config or {}
    rows = (await ctx.db.execute(select(Channel.config).where(Channel.platform == "instagram", Channel.id != channel.id))).scalars()
    load: dict[int, int] = {}
    for c in rows:
        if (c or {}).get("proxy_node_id"):
            load[c["proxy_node_id"]] = load.get(c["proxy_node_id"], 0) + 1
    route = proxy_pool.pool.sticky("instagram", cfg.get("proxy_node_id"), load, tuple(cfg.get("bad_nodes") or ()),
                                   channel_id=channel.id)
    if cfg.get("own_proxy") and (route is None or route[0] != cfg.get("proxy_node_id")):
        # the seller's own proxy only: never fall back to the shared ones
        raise PlatformError("پروکسی اختصاصی این حساب (IP ثابت خودت) کار نمی‌کند؛ در صفحهٔ کانال‌ها «تست پروکسی» یا «تغییر پروکسی» را بزن.",
                            "own_proxy_down", RECONNECT)
    if route is None:
        raise PlatformError("هیچ پروکسی سالمی برای اینستاگرام در پنل ادمین › پروکسی‌ها نیست؛ ادمین باید پروکسی اضافه کند.",
                            "no_proxy", PROXY_HELP)
    node_id, url = route
    if node_id != cfg.get("proxy_node_id"):
        channel.config = {**cfg, "proxy_node_id": node_id}
    return _socks(url)


def _client(proxy: str, saved: dict[str, Any] | None):
    from instagrapi import Client

    cl = Client()
    cl.delay_range = [2, 6]  # human-like pauses between the calls one action makes
    if saved:
        cl.set_settings(saved)  # same device ids, user agent and cookies as last time
    cl.set_locale("fa_IR")
    cl.set_country("IR")
    cl.set_timezone_offset(int(3.5 * 3600))
    if proxy:
        cl.set_proxy(proxy)
    return cl


async def _in_thread(fn: Callable, *args, **kw):
    return await asyncio.to_thread(fn, *args, **kw)


class PrivateInstagram:
    """Used by the instagram adapter for channels connected with username + password (config.mode == "private")."""

    # ---------- connect ----------

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        username = (data.get("username") or "").strip().lstrip("@").lower()
        password = data.get("password") or ""
        if not username or not password:
            raise PlatformError("نام کاربری و رمز اینستاگرام را وارد کنید.", "missing_login")
        old = decrypt_credentials(channel.credentials_encrypted) if channel.credentials_encrypted else {}
        saved = old.get("settings") if old.get("username") == username else None
        proxy = await _route(ctx, channel)
        loop = asyncio.get_running_loop()
        cl = _client(proxy, saved)
        wait = _CodeWait()

        def ask(kind: str) -> str:
            """Runs in the login thread: tell the seller a code is needed, then wait for it."""
            asyncio.run_coroutine_threadsafe(self._need_code(ctx, kind), loop).result(30)
            wait.event.clear()
            if not wait.event.wait(CODE_WAIT_SECONDS):
                return ""
            return wait.code

        def challenge_code(_username, choice) -> str:
            return ask("email" if "mail" in str(choice).lower() else "sms")

        cl.challenge_code_handler = challenge_code

        def login() -> None:
            from instagrapi.exceptions import TwoFactorRequired

            try:
                cl.login(username, password)
            except TwoFactorRequired:
                code = ask("2fa")
                if not code:
                    raise
                cl.login(username, password, verification_code=code)

        _waits[channel.id] = wait
        try:
            async with ctx.step("login", "ورود به اینستاگرام", request={"username": username, "via_proxy": bool(proxy)}) as s:
                try:
                    await _in_thread(login)
                except Exception as exc:  # noqa: BLE001
                    raise _humanize(exc) from exc
                s.response = {"user_id": str(cl.user_id)}
        finally:
            _waits.pop(channel.id, None)
            ctx.op.result = {}
        async with ctx.step("account", "خواندن اطلاعات حساب") as s:
            try:
                me = await _in_thread(cl.account_info)
            except Exception as exc:  # noqa: BLE001
                raise _humanize(exc) from exc
            s.response = {"username": me.username, "is_business": getattr(me, "is_business", None)}
        cfg = {k: v for k, v in (channel.config or {}).items() if k not in ("paused_until", "pause_reason")}
        cfg.update(mode="private", ig_user_id=str(cl.user_id), username=me.username, account_type="private_api")
        channel.config = cfg
        return {"credentials": {"username": username, "password": password, "settings": cl.get_settings()},
                "config": cfg, "display_name": "@" + me.username, "external_account_id": str(cl.user_id)}

    async def _need_code(self, ctx, kind: str) -> None:
        label = {"2fa": "کد تأیید دومرحله‌ای (اپ احراز هویت یا پیامک)", "sms": "کدی که اینستاگرام پیامک کرد",
                 "email": "کدی که اینستاگرام ایمیل کرد"}[kind]
        ctx.op.result = {"needs_code": kind, "code_label": label}
        ctx.op.current_step = "منتظر کد تأیید اینستاگرام"
        await ctx.db.commit()

    # ---------- session ----------

    async def _session(self, ctx, channel: Channel):
        creds = decrypt_credentials(channel.credentials_encrypted)
        if not creds.get("settings"):
            raise PlatformError("حساب اینستاگرام وصل نیست.", "missing_token", RECONNECT)
        proxy = await _route(ctx, channel)
        return _client(proxy, creds["settings"]), creds

    def _save(self, channel: Channel, cl, creds: dict[str, Any]) -> None:
        channel.credentials_encrypted = encrypt_credentials({**creds, "settings": cl.get_settings()})

    async def _call(self, ctx, channel: Channel, cl, creds: dict[str, Any], fn: Callable, *args, **kw):
        """One instagrapi call; when the session expired, one quiet re-login (same device) and one retry."""
        from instagrapi.exceptions import LoginRequired

        try:
            return await _in_thread(fn, *args, **kw)
        except LoginRequired:
            if creds.get("relogin_at") and time.time() - creds["relogin_at"] < 6 * 3600:
                raise _humanize(LoginRequired(), channel)  # never re-login in a loop
            creds["relogin_at"] = time.time()
            async with ctx.step("relogin", "ورود دوباره به اینستاگرام (نشست قبلی تمام شده بود)"):
                try:
                    await _in_thread(cl.relogin)
                except Exception as exc:  # noqa: BLE001
                    raise _humanize(exc, channel) from exc
            self._save(channel, cl, creds)
            try:
                return await _in_thread(fn, *args, **kw)
            except Exception as exc:  # noqa: BLE001
                raise _humanize(exc, channel) from exc
        except Exception as exc:  # noqa: BLE001
            raise _humanize(exc, channel) from exc

    async def health(self, ctx, channel: Channel) -> None:
        cl, creds = await self._session(ctx, channel)
        async with ctx.step("account", "بررسی نشست اینستاگرام") as s:
            me = await self._call(ctx, channel, cl, creds, cl.account_info)
            s.response = {"username": me.username}
        self._save(channel, cl, creds)

    # ---------- publish ----------

    def validate(self, product: Product, fmt: str) -> list[str]:
        if fmt == "reel" and not product.video_url:
            return ["برای ریلز، محصول باید ویدیو داشته باشد؛ در صفحهٔ محصول ویدیو آپلود کن."]
        if fmt in ("post", "story") and not product.images and not product.video_url:
            return ["اینستاگرام حداقل یک عکس یا ویدیو لازم دارد."]
        if fmt == "post" and not product.images:
            return ["پست اینستاگرام حداقل یک عکس لازم دارد (برای ویدیو، ریلز را انتخاب کن)."]
        return []

    async def publish(self, ctx, channel: Channel, product: Product, store: Store, fmt: str) -> PublishResult:
        cl, creds = await self._session(ctx, channel)
        payload = getattr(getattr(ctx, "op", None), "payload", None) or {}
        caption = caption_for(product, store.settings or {}, show_price=payload.get("show_price"))[:CAPTION_LIMIT]
        with tempfile.TemporaryDirectory(prefix="ig-") as tmp:
            folder = Path(tmp)
            if fmt == "reel" or (fmt == "story" and product.video_url):
                async with ctx.step("video", "آماده‌سازی ویدیو (MP4)") as s:
                    src = folder / ("src" + Path(product.video_url.split("?")[0]).suffix)
                    src.write_bytes(await load_bytes(product.video_url, max_bytes=video.MAX_VIDEO_BYTES))
                    mp4, thumb = await video.for_instagram(src, folder, video.STORY_SECONDS if fmt == "story" else video.MAX_SECONDS,
                                                           vertical=fmt == "story")
                    s.response = {"bytes": mp4.stat().st_size}
                label = "انتشار ریلز" if fmt == "reel" else "انتشار استوری ویدیویی"
                async with ctx.step("publish", label) as s:
                    if fmt == "reel":
                        media = await self._call(ctx, channel, cl, creds, cl.clip_upload, mp4, caption, thumbnail=thumb)
                    else:
                        media = await self._call(ctx, channel, cl, creds, cl.video_upload_to_story, mp4, thumbnail=thumb)
                    s.response = {"pk": str(media.pk)}
            else:
                async with ctx.step("images", "آماده‌سازی عکس‌ها (JPEG)") as s:
                    paths = []
                    for i, img in enumerate(product.images[:1 if fmt == "story" else MAX_ALBUM]):
                        p = folder / f"{i}.jpg"
                        raw = await load_bytes(img.url)
                        p.write_bytes(story_frame(raw) if fmt == "story" else to_jpeg(raw, max_side=1440))
                        paths.append(p)
                    s.response = {"images": len(paths)}
                label = {"story": "انتشار استوری", "post": "انتشار پست" if len(paths) == 1 else f"انتشار آلبوم {len(paths)} عکسی"}[fmt]
                async with ctx.step("publish", label) as s:
                    if fmt == "story":
                        media = await self._call(ctx, channel, cl, creds, cl.photo_upload_to_story, paths[0])
                    elif len(paths) == 1:
                        media = await self._call(ctx, channel, cl, creds, cl.photo_upload, paths[0], caption)
                    else:
                        media = await self._call(ctx, channel, cl, creds, cl.album_upload, paths, caption)
                    s.response = {"pk": str(media.pk)}
        self._save(channel, cl, creds)
        code = getattr(media, "code", "") or ""
        username = (channel.config or {}).get("username", "")
        if fmt == "story":
            url = f"https://www.instagram.com/stories/{username}/{media.pk}/" if username else ""
        elif code:
            url = f"https://www.instagram.com/{'reel' if fmt == 'reel' else 'p'}/{code}/"
        else:
            url = ""
        return PublishResult(str(media.pk), url)

    async def fetch_items(self, ctx, channel: Channel, limit: int = 50) -> list:
        from app.platforms.base import CrawledItem

        cl, creds = await self._session(ctx, channel)
        async with ctx.step("media", "دریافت پست‌های حساب") as s:
            medias = await self._call(ctx, channel, cl, creds, cl.user_medias, int(channel.config["ig_user_id"]), min(limit, 50))
            s.response = {"count": len(medias)}
        self._save(channel, cl, creds)
        out = []
        for m in medias:
            imgs = [str(r.thumbnail_url) for r in (m.resources or []) if r.thumbnail_url and r.media_type == 1] or (
                [str(m.thumbnail_url)] if m.thumbnail_url and m.media_type == 1 else [])
            out.append(CrawledItem(external_id=str(m.pk), caption=m.caption_text or "", image_urls=imgs,
                                   url=f"https://www.instagram.com/p/{m.code}/", raw={"pk": str(m.pk), "code": m.code}))
        return out


private = PrivateInstagram()
