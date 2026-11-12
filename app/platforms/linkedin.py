"""LinkedIn through its own web API (Voyager, the API www.linkedin.com uses): the seller signs in with email and
password, or pastes the `li_at` cookie from a browser where they are signed in.

LinkedIn's public API only lets an app post to a profile; reading comments and messages is closed to apps, and a
LinkedIn developer app is hard to get from Iran. So, as with Instagram, the account is used the way its owner's
browser uses it, and everything is built to look like that one browser:
- one sign-in, then the saved cookies and the same user agent are reused; a fresh sign-in only when LinkedIn drops
  the session, and never in a loop;
- every account goes through one fixed proxy from the admin's pool (admin panel › proxies, "LinkedIn");
- posts are paced: a random gap between two posts of one account and a daily cap (admin panel settings);
- when LinkedIn pushes back (a security check, 429, "restricted"), the account is paused and the seller is told.

LinkedIn changes this API without notice, so every response is read defensively, and the messaging GraphQL ids can
be replaced from the admin panel."""

import asyncio
import json
import logging
import random
import re
import time
import uuid
from datetime import timedelta
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select

from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, PlatformError, PublishResult
from app.services import proxy_pool
from app.services.images import load_bytes, to_jpeg
from app.services.integrations import linkedin_app
from app.services.text import caption_for

log = logging.getLogger("sello.linkedin")

BASE = "https://www.linkedin.com"
API = BASE + "/voyager/api"
TEXT_LIMIT = 3000
MAX_IMAGES = 9
PAUSE_HOURS = {"limit": 6, "restricted": 48, "challenge": 0}
PROXY_HELP = {"type": "help", "label": "پروکسی لینکدین در پنل ادمین"}
RETRY = {"type": "retry", "label": "تلاش دوباره"}
COOKIE = {"type": "cookie", "label": "اتصال با کوکی مرورگر"}
# One desktop browser per account, picked at connect and kept (a new user agent every call looks like a bot).
USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:144.0) Gecko/20100101 Firefox/144.0",
)
AUTH_HEADERS = {"X-Li-User-Agent": "LIAuthLibrary:0.0.3 com.linkedin.android:4.1.881 Asus_ASUS_Z01QD:android_9",
                "User-Agent": "ANDROID OS", "X-User-Language": "en", "X-User-Locale": "en_US",
                "Accept-Language": "en-us"}
# Messaging GraphQL query ids the web app used when this was written; the admin can override them.
QUERY_IDS = {
    "conversations": "messengerConversations.0d5e6781bbee71c3e51c8843c6519f48",
    "messages": "messengerMessages.5846eeb71c981f11e0134cb6626cc314",
}


def _socks(url: str) -> str:
    """Resolve names through the proxy too: LinkedIn's domains are filtered in Iran's DNS."""
    return "socks5h://" + url.removeprefix("socks5://") if url.startswith("socks5://") else url


def find_urns(data: Any, kind: str) -> list[str]:
    """Every `urn:li:<kind>:<id>` in a response, in order (LinkedIn nests them differently from week to week)."""
    text = json.dumps(data) if not isinstance(data, str) else data
    return list(dict.fromkeys(re.findall(rf"urn:li:{kind}:[0-9A-Za-z_-]+", text)))


def dig(data: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return default
        data = data[k]
    return data


def post_url(urn: str) -> str:
    return f"{BASE}/feed/update/{urn}/" if urn else ""


# ---------- pausing ----------

def pause(channel: Channel, kind: str, reason: str) -> None:
    cfg = dict(channel.config or {})
    hours = PAUSE_HOURS.get(kind, 24)
    if hours:
        cfg["paused_until"] = (utcnow() + timedelta(hours=hours)).isoformat()
    cfg["pause_reason"] = reason
    channel.config = cfg


def _status_error(resp: httpx.Response, channel: Channel | None) -> PlatformError:
    code = resp.status_code
    body = resp.text[:300]
    if code in (401, 403) and ("CSRF" in body or code == 401):
        return PlatformError("نشست لینکدین تمام شده؛ دوباره وصل شو.", "expired", RECONNECT, body)
    if code == 999 or "checkpoint" in str(resp.headers.get("location", "")) or "challenge" in body.lower():
        msg = ("لینکدین تأیید امنیتی خواست. در مرورگر وارد لینکدین شو، اگر پیامی آمد تأیید کن، بعد دوباره وصل شو "
               "(یا کوکی li_at همان مرورگر را بده).")
        if channel:
            pause(channel, "challenge", msg)
        return PlatformError(msg, "li_challenge", COOKIE, body)
    if code == 429:
        msg = "لینکدین گفت کمی صبر کنید؛ کارهای این حساب شش ساعت عقب افتاد."
        if channel:
            pause(channel, "limit", msg)
        return PlatformError(msg, "li_wait", RETRY, body)
    if code == 403 and "restrict" in body.lower():
        msg = "لینکدین این حساب را موقتاً محدود کرد؛ ۴۸ ساعت کاری انجام نمی‌شود. در مرورگر وارد لینکدین شو و هشدار را ببین."
        if channel:
            pause(channel, "restricted", msg)
        return PlatformError(msg, "li_restricted", RECONNECT, body)
    if code == 403:
        return PlatformError("لینکدین اجازهٔ این کار را نداد (۴۰۳).", "permission", RECONNECT, body)
    return PlatformError(f"لینکدین خطا داد ({code}).", "platform_error", detail=body)


# ---------- the client ----------

class Session:
    """One signed-in browser: cookies, csrf token and user agent, through the account's fixed proxy."""

    def __init__(self, proxy: str, cookies: dict[str, str], user_agent: str, channel: Channel | None = None):
        self.proxy = proxy
        self.cookies = dict(cookies)
        self.user_agent = user_agent
        self.channel = channel

    @property
    def csrf(self) -> str:
        return (self.cookies.get("JSESSIONID") or "").strip('"')

    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return {"user-agent": self.user_agent, "accept": "application/vnd.linkedin.normalized+json+2.1",
                "accept-language": "en-US,en;q=0.9,fa;q=0.8", "x-li-lang": "en_US",
                "x-restli-protocol-version": "2.0.0", "csrf-token": self.csrf, **(extra or {})}

    def _client(self, timeout: float = 60) -> httpx.AsyncClient:
        t = httpx.Timeout(timeout, connect=min(proxy_pool.CONNECT_TIMEOUT, timeout)) if self.proxy else httpx.Timeout(timeout)
        return httpx.AsyncClient(timeout=t, proxy=self.proxy or None, follow_redirects=False)

    async def raw(self, method: str, url: str, *, headers: dict[str, str] | None = None, **kw) -> httpx.Response:
        try:
            async with self._client() as c:
                c.cookies.update(self.cookies)
                resp = await c.request(method, url, headers=headers if headers is not None else self.headers(), **kw)
        except httpx.HTTPError as exc:
            raise PlatformError("ارتباط با لینکدین از طریق پروکسی برقرار نشد؛ پروکسی‌های لینکدین را در پنل ادمین بررسی کن.",
                                "network", PROXY_HELP, str(exc)[:300]) from exc
        for name, value in resp.cookies.items():
            if value and value != "delete me":
                self.cookies[name] = value
        return resp

    async def api(self, method: str, path: str, *, ok: tuple[int, ...] = (200, 201), **kw) -> Any:
        resp = await self.raw(method, path if path.startswith("http") else API + path, **kw)
        if resp.status_code not in ok:
            raise _status_error(resp, self.channel)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            raise PlatformError("لینکدین پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])

    # ---------- account ----------

    async def me(self) -> dict[str, Any]:
        data = await self.api("GET", "/me")
        mini = {}
        for item in data.get("included") or []:
            if "publicIdentifier" in item or "firstName" in item:
                mini = item
                break
        mini = mini or dig(data, "miniProfile") or dig(data, "data", "miniProfile") or {}
        urn = (mini.get("dashEntityUrn") or mini.get("entityUrn") or dig(data, "data", "*miniProfile") or "")
        profile_id = urn.rsplit(":", 1)[-1] if urn else ""
        plain = dig(data, "data", "plainId") or dig(data, "plainId") or ""
        name = " ".join(x for x in (mini.get("firstName"), mini.get("lastName")) if x).strip()
        if not profile_id:
            raise PlatformError("اطلاعات حساب لینکدین خوانده نشد؛ نشست معتبر نیست.", "expired", RECONNECT, data)
        return {"profile_id": profile_id, "member_id": str(plain), "public_id": mini.get("publicIdentifier") or "",
                "name": name}

    # ---------- posts ----------

    async def upload_image(self, data: bytes, filename: str) -> str:
        meta = await self.api("POST", "/voyagerVideoDashMediaUploadMetadata?action=upload",
                              json={"mediaUploadType": "IMAGE_SHARING", "fileSize": len(data), "filename": filename})
        value = dig(meta, "data", "value") or dig(meta, "value") or {}
        upload_url = value.get("singleUploadUrl") or ""
        urn = value.get("urn") or next(iter(find_urns(meta, "digitalmediaAsset")), "")
        if not upload_url or not urn:
            raise PlatformError("لینکدین آدرس آپلود عکس نداد.", "bad_response", detail=meta)
        resp = await self.raw("PUT", upload_url, content=data,
                              headers=self.headers({"content-type": "image/jpeg", "media-type-family": "STILLIMAGE"}))
        if resp.status_code not in (200, 201):
            raise PlatformError("آپلود عکس در لینکدین ناموفق بود.", "bad_image", EDIT_PRODUCT, resp.text[:300])
        return urn

    async def create_post(self, text: str, media_urns: list[str]) -> str:
        body = {"visibleToConnectionsOnly": False, "externalAudienceProviders": [],
                "commentaryV2": {"text": text, "attributes": []}, "origin": "FEED",
                "allowedCommentersScope": "ALL", "postState": "PUBLISHED",
                "media": [{"category": "IMAGE", "mediaUrn": u, "tapTargets": []} for u in media_urns]}
        data = await self.api("POST", "/contentcreation/normShares", json=body)
        urn = next(iter(find_urns(data, "activity")), "") or next(iter(find_urns(data, "share")), "") \
            or next(iter(find_urns(data, "ugcPost")), "")
        if not urn:
            raise PlatformError("لینکدین پست را ساخت ولی شناسه‌اش را نداد.", "bad_response", detail=str(data)[:300])
        return urn

    async def comments(self, activity_urn: str, count: int = 20) -> list[dict[str, Any]]:
        """The newest top-level comments of one post: {id, urn, text, author_id, author_name, created_at (s)}."""
        act = activity_urn.rsplit(":", 1)[-1]
        data = await self.api("GET", "/feed/comments", headers=self.headers({"accept": "application/json"}),
                              params={"count": count, "start": 0, "q": "comments", "sortOrder": "REVERSE_CHRONOLOGICAL",
                                      "updateId": f"activity:{act}"})
        out = []
        for el in data.get("elements") or []:
            urn = el.get("urn") or el.get("dashEntityUrn") or ""
            m = re.search(r"\(([^,]+),(\d+)\)", el.get("entityUrn") or urn)
            cid = m.group(2) if m else urn.rsplit(",", 1)[-1].rstrip(")")
            values = dig(el, "comment", "values")
            text = dig(el, "commentV2", "text") or (values[0].get("value") if isinstance(values, list) and values else "") or ""
            actor = dig(el, "commenter") or {}
            member = next((v for k, v in actor.items() if isinstance(v, dict) and "miniProfile" in v), {}) or actor
            mini = member.get("miniProfile") or {}
            out.append({"id": str(cid), "urn": urn or f"urn:li:comment:(activity:{act},{cid})", "text": text,
                        "author_id": (mini.get("entityUrn") or "").rsplit(":", 1)[-1],
                        "author_public_id": mini.get("publicIdentifier") or "",
                        "author_name": " ".join(x for x in (mini.get("firstName"), mini.get("lastName")) if x).strip(),
                        "created_at": int(el.get("createdTime") or 0) / 1000,
                        "is_reply": bool(el.get("parentCommentUrn") or el.get("parentComment"))})
        return out

    async def reply_comment(self, activity_urn: str, comment_id: str, text: str) -> None:
        act = activity_urn if activity_urn.startswith("urn:") else f"urn:li:activity:{activity_urn}"
        await self.api("POST", "/voyagerSocialDashNormComments",
                       json={"commentary": {"text": text, "attributesV2": [],
                                            "$type": "com.linkedin.voyager.dash.common.text.TextViewModel"},
                             "threadUrn": f"urn:li:comment:({act},{comment_id})"})

    # ---------- messages ----------

    async def conversations(self, profile_id: str, query_ids: dict[str, str]) -> list[dict[str, Any]]:
        """Recent 1:1 conversations with their last messages:
        [{urn, last_activity (s), other: {id, name, public_id}, messages: [{id, text, sender_id, at (s)}]}]."""
        qid = query_ids.get("conversations") or QUERY_IDS["conversations"]
        mailbox = quote(f"urn:li:fsd_profile:{profile_id}", safe="")
        data = await self.api("GET", f"/voyagerMessagingGraphQL/graphql?queryId={qid}&variables=(mailboxUrn:{mailbox})",
                              headers=self.headers({"accept": "application/graphql"}))
        root = dig(data, "data") or {}
        block = next((v for k, v in root.items() if k.startswith("messengerConversations") and isinstance(v, dict)), {})
        out = []
        for c in block.get("elements") or []:
            if c.get("groupChat"):
                continue
            others = []
            for p in c.get("conversationParticipants") or []:
                pid = (p.get("hostIdentityUrn") or "").rsplit(":", 1)[-1]
                if pid and pid != profile_id:
                    member = dig(p, "participantType", "member") or {}
                    others.append({"id": pid, "public_id": (member.get("profileUrl") or "").rstrip("/").rsplit("/", 1)[-1],
                                   "name": " ".join(x for x in (dig(member, "firstName", "text"), dig(member, "lastName", "text")) if x)})
            if len(others) != 1:
                continue
            msgs = []
            for m in dig(c, "messages", "elements", default=[]) or []:
                msgs.append({"id": m.get("entityUrn") or m.get("backendUrn") or "",
                             "text": dig(m, "body", "text") or "",
                             "sender_id": (dig(m, "sender", "hostIdentityUrn") or "").rsplit(":", 1)[-1],
                             "at": int(m.get("deliveredAt") or 0) / 1000,
                             "has_media": bool(m.get("renderContent"))})
            out.append({"urn": c.get("entityUrn") or "", "last_activity": int(c.get("lastActivityAt") or 0) / 1000,
                        "other": others[0], "messages": msgs})
        return out

    async def messages(self, conversation_urn: str, query_ids: dict[str, str]) -> list[dict[str, Any]]:
        qid = query_ids.get("messages") or QUERY_IDS["messages"]
        data = await self.api("GET", f"/voyagerMessagingGraphQL/graphql?queryId={qid}"
                                     f"&variables=(conversationUrn:{quote(conversation_urn, safe='')})",
                              headers=self.headers({"accept": "application/graphql"}))
        root = dig(data, "data") or {}
        block = next((v for k, v in root.items() if k.startswith("messengerMessages") and isinstance(v, dict)), {})
        return [{"id": m.get("entityUrn") or "", "text": dig(m, "body", "text") or "",
                 "sender_id": (dig(m, "sender", "hostIdentityUrn") or "").rsplit(":", 1)[-1],
                 "at": int(m.get("deliveredAt") or 0) / 1000, "has_media": bool(m.get("renderContent"))}
                for m in block.get("elements") or []]

    async def send_message(self, profile_id: str, text: str, *, conversation_urn: str = "",
                           recipient_id: str = "") -> dict[str, str]:
        """Sends a DM into a conversation, or to a member (a new conversation). Returns {id, conversation}."""
        mailbox = f"urn:li:fsd_profile:{profile_id}"
        message: dict[str, Any] = {"body": {"attributes": [], "text": text}, "renderContentUnions": [],
                                   "originToken": str(uuid.uuid4())}
        body: dict[str, Any] = {"message": message, "mailboxUrn": mailbox, "dedupeByClientGeneratedToken": False,
                                "trackingId": "".join(chr(random.randrange(256)) for _ in range(16))}
        if conversation_urn:
            message["conversationUrn"] = conversation_urn
        else:
            body["hostRecipientUrns"] = [f"urn:li:fsd_profile:{recipient_id}"]
        data = await self.api("POST", "/voyagerMessagingDashMessengerMessages?action=createMessage", json=body,
                              headers=self.headers({"accept": "application/json", "content-type": "text/plain;charset=UTF-8"}))
        value = dig(data, "value") or dig(data, "data", "value") or {}
        conv = value.get("conversationUrn") or next(iter(find_urns(data, "msg_conversation")), "") or conversation_urn
        return {"id": value.get("entityUrn") or value.get("backendUrn") or "", "conversation": conv}


# ---------- sign in ----------

async def sign_in(proxy: str, email: str, password: str, user_agent: str) -> dict[str, str]:
    """Email + password, the way LinkedIn's Android app signs in. Returns the session cookies."""
    s = Session(proxy, {}, user_agent)
    first = await s.raw("GET", BASE + "/uas/authenticate", headers=AUTH_HEADERS)
    if first.status_code >= 500:
        raise PlatformError("لینکدین در دسترس نیست؛ کمی بعد دوباره تلاش کن.", "network", RETRY)
    if not s.csrf:
        raise PlatformError("لینکدین صفحهٔ ورود را نداد؛ پروکسی لینکدین را بررسی کن.", "network", PROXY_HELP)
    resp = await s.raw("POST", BASE + "/uas/authenticate", headers=AUTH_HEADERS,
                       data={"session_key": email, "session_password": password, "JSESSIONID": s.cookies["JSESSIONID"]})
    try:
        result = (resp.json() or {}).get("login_result", "")
    except ValueError:
        result = ""
    if result == "PASS" and s.cookies.get("li_at"):
        return s.cookies
    if result in ("BAD_PASSWORD", "BAD_USERNAME_OR_PASSWORD"):
        raise PlatformError("رمز لینکدین اشتباه است.", "bad_password", RECONNECT)
    if result in ("BAD_EMAIL", "BAD_USERNAME"):
        raise PlatformError("این ایمیل یا شماره در لینکدین پیدا نشد.", "user_not_found", RECONNECT)
    if result == "CHALLENGE" or resp.status_code in (303, 999):
        raise PlatformError(
            "لینکدین برای این ورود تأیید امنیتی خواست که از اینجا ممکن نیست. یک بار در مرورگر وارد لینکدین شو، "
            "بعد مقدار کوکی li_at را از همان مرورگر کپی کن و در «اتصال با کوکی» بگذار.", "li_challenge", COOKIE, result)
    if resp.status_code == 429:
        raise PlatformError("لینکدین گفت کمی صبر کنید؛ چند دقیقهٔ دیگر دوباره تلاش کن.", "li_wait", RETRY)
    raise PlatformError(f"ورود به لینکدین ناموفق بود ({result or resp.status_code}).", "login_failed", RECONNECT,
                        resp.text[:300])


async def from_cookie(proxy: str, li_at: str, user_agent: str) -> dict[str, str]:
    """A pasted `li_at` cookie; LinkedIn hands out the JSESSIONID (csrf) on the first page load."""
    li_at = li_at.strip().strip('"').removeprefix("li_at=").split(";")[0].strip()
    if len(li_at) < 20:
        raise PlatformError("کوکی li_at درست نیست؛ کل مقدارش را کپی کن.", "bad_cookie", COOKIE)
    s = Session(proxy, {"li_at": li_at}, user_agent)
    await s.raw("GET", BASE + "/feed/", headers={"user-agent": user_agent, "accept": "text/html"})
    if not s.csrf:
        s.cookies["JSESSIONID"] = f'"ajax:{random.randrange(10**18):019d}"'
    return s.cookies


# ---------- the adapter ----------

_route_lock = asyncio.Lock()


async def _route(ctx, channel: Channel) -> str:
    """The account's own proxy node (kept while it is healthy), else the one fewest LinkedIn accounts use."""
    cfg = channel.config or {}
    rows = (await ctx.db.execute(select(Channel.config).where(Channel.platform == "linkedin", Channel.id != channel.id))).scalars()
    load: dict[int, int] = {}
    for c in rows:
        if (c or {}).get("proxy_node_id"):
            load[c["proxy_node_id"]] = load.get(c["proxy_node_id"], 0) + 1
    route = proxy_pool.pool.sticky("linkedin", cfg.get("proxy_node_id"), load, tuple(cfg.get("bad_nodes") or ()))
    if route is None:
        raise PlatformError("هیچ پروکسی سالمی برای لینکدین در پنل ادمین › پروکسی‌ها نیست؛ ادمین باید پروکسی اضافه کند.",
                            "no_proxy", PROXY_HELP)
    node_id, url = route
    if node_id != cfg.get("proxy_node_id"):
        channel.config = {**cfg, "proxy_node_id": node_id}
    return _socks(url)


class LinkedInAdapter:
    platform = "linkedin"
    label_fa = "لینکدین"

    async def _app(self, ctx) -> dict[str, Any]:
        app = await linkedin_app(ctx.db)
        if not app["enabled"]:
            raise PlatformError("اتصال لینکدین هنوز توسط ادمین فعال نشده است.", "disabled", {"type": "info", "label": "به‌زودی"})
        return app

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        await self._app(ctx)
        email = (data.get("email") or data.get("username") or "").strip().lower()
        password = data.get("password") or ""
        li_at = (data.get("li_at") or data.get("cookie") or "").strip()
        if not li_at and not (email and password):
            raise PlatformError("ایمیل و رمز لینکدین، یا کوکی li_at را وارد کنید.", "missing_login")
        old = decrypt_credentials(channel.credentials_encrypted) if channel.credentials_encrypted else {}
        ua = old.get("user_agent") or random.choice(USER_AGENTS)
        proxy = await _route(ctx, channel)
        async with ctx.step("login", "ورود به لینکدین" + (" با کوکی" if li_at else ""),
                            request={"email": email, "cookie": bool(li_at), "via_proxy": bool(proxy)}) as s:
            cookies = await (from_cookie(proxy, li_at, ua) if li_at else sign_in(proxy, email, password, ua))
            s.response = {"cookies": sorted(cookies)}
        session = Session(proxy, cookies, ua, channel)
        async with ctx.step("account", "خواندن اطلاعات حساب") as s:
            me = await session.me()
            s.response = me
        cfg = {k: v for k, v in (channel.config or {}).items() if k not in ("paused_until", "pause_reason")}
        cfg.update(mode="session", **me)
        channel.config = cfg
        creds = {"email": email, "password": password if not li_at else old.get("password", ""),
                 "cookies": session.cookies, "user_agent": ua, "by_cookie": bool(li_at)}
        return {"credentials": creds, "config": cfg, "display_name": me["name"] or me["public_id"] or "LinkedIn",
                "external_account_id": me["profile_id"]}

    async def session(self, ctx, channel: Channel) -> tuple[Session, dict[str, Any]]:
        creds = decrypt_credentials(channel.credentials_encrypted)
        if not creds.get("cookies"):
            raise PlatformError("حساب لینکدین وصل نیست.", "missing_token", RECONNECT)
        proxy = await _route(ctx, channel)
        return Session(proxy, creds["cookies"], creds.get("user_agent") or USER_AGENTS[0], channel), creds

    def save(self, channel: Channel, session: Session, creds: dict[str, Any]) -> None:
        if session.cookies != creds.get("cookies"):
            channel.credentials_encrypted = encrypt_credentials({**creds, "cookies": session.cookies})

    async def call(self, ctx, channel: Channel, session: Session, creds: dict[str, Any], fn, *args, **kw):
        """One API call; when the session expired and the account has a password, one quiet sign-in and one retry."""
        try:
            return await fn(*args, **kw)
        except PlatformError as exc:
            if exc.code != "expired" or not (creds.get("email") and creds.get("password")):
                raise
            if creds.get("relogin_at") and time.time() - creds["relogin_at"] < 6 * 3600:
                raise  # never sign in again and again
            creds["relogin_at"] = time.time()
            async with ctx.step("relogin", "ورود دوباره به لینکدین (نشست قبلی تمام شده بود)"):
                session.cookies = await sign_in(session.proxy, creds["email"], creds["password"], session.user_agent)
            channel.credentials_encrypted = encrypt_credentials({**creds, "cookies": session.cookies})
            return await fn(*args, **kw)

    async def health(self, ctx, channel: Channel) -> None:
        await self._app(ctx)
        session, creds = await self.session(ctx, channel)
        async with ctx.step("account", "بررسی نشست لینکدین") as s:
            s.response = await self.call(ctx, channel, session, creds, session.me)
        self.save(channel, session, creds)

    def validate(self, product: Product) -> list[str]:
        if not (product.title or "").strip():
            return ["پست لینکدین متن لازم دارد."]
        return []

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        await self._app(ctx)
        session, creds = await self.session(ctx, channel)
        payload = getattr(getattr(ctx, "op", None), "payload", None) or {}
        text = caption_for(product, store.settings or {}, show_price=payload.get("show_price"))[:TEXT_LIMIT]
        urns: list[str] = []
        if product.images:
            async with ctx.step("images", "آپلود عکس‌ها در لینکدین") as s:
                for i, img in enumerate(product.images[:MAX_IMAGES]):
                    if i:
                        await asyncio.sleep(random.uniform(1, 3))
                    data = to_jpeg(await load_bytes(img.url), max_side=2048)
                    urns.append(await self.call(ctx, channel, session, creds, session.upload_image, data, f"product-{i + 1}.jpg"))
                s.response = {"images": len(urns)}
        async with ctx.step("publish", "انتشار پست" if len(urns) < 2 else f"انتشار پست با {len(urns)} عکس") as s:
            urn = await self.call(ctx, channel, session, creds, session.create_post, text, urns)
            s.response = {"urn": urn}
        self.save(channel, session, creds)
        return PublishResult(urn, post_url(urn))

    async def fetch_items(self, ctx, channel: Channel, limit: int = 50) -> list:
        raise PlatformError("خواندن محصول از لینکدین پشتیبانی نمی‌شود.", "not_supported")


linkedin = LinkedInAdapter()
