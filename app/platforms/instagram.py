"""Instagram through the official "Instagram API with Instagram Login" (graph.instagram.com).

Requirements (Meta docs, content-publishing): a Professional (Business or Creator) account, a Meta app
with instagram_business_basic + instagram_business_content_publish (Advanced access needs App Review),
JPEG images on a public URL (Instagram downloads them itself), at most 10 items per carousel, and
100 API-published posts per rolling 24 hours. graph.instagram.com is filtered in Iran, so the admin can
set a proxy. The Meta app id/secret are entered in the admin panel; sellers connect with the OAuth
button, or paste a long-lived token as a fallback."""

import asyncio
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from app.core.config import settings
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, CrawledItem, PlatformError, PublishResult
from app.services.images import public_jpeg
from app.services.integrations import INSTAGRAM_SCOPES, instagram_app
from app.services.text import caption_for

GRAPH = "https://graph.instagram.com"
OAUTH_AUTHORIZE = "https://www.instagram.com/oauth/authorize"
OAUTH_TOKEN = "https://api.instagram.com/oauth/access_token"
MAX_CAROUSEL = 10
CAPTION_LIMIT = 2200
REFRESH_BEFORE = 10 * 86400  # refresh long-lived tokens (60 days) when fewer than 10 days remain
SETTINGS_HELP = {"type": "help", "label": "تنظیمات اینستاگرام در پنل ادمین"}


def authorize_url(app: dict[str, Any], state: str) -> str:
    return OAUTH_AUTHORIZE + "?" + urlencode({
        "client_id": app["app_id"], "redirect_uri": app["redirect_uri"], "response_type": "code",
        "scope": INSTAGRAM_SCOPES, "state": state, "enable_fb_login": "0", "force_authentication": "1"})


class InstagramAdapter:
    platform = "instagram"
    label_fa = "اینستاگرام"
    poll_seconds = 3.0

    async def _app(self, ctx) -> dict[str, Any]:
        app = await instagram_app(ctx.db)
        if not app["enabled"]:
            raise PlatformError("اتصال اینستاگرام هنوز توسط ادمین فعال نشده است.", "disabled",
                                {"type": "info", "label": "به‌زودی"})
        return app

    async def _http(self, app: dict[str, Any], method: str, url: str, **kw) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=60, proxy=app.get("proxy") or None) as client:
                resp = await client.request(method, url, **kw)
        except httpx.HTTPError as exc:
            hint = "" if app.get("proxy") else " دامنه‌های اینستاگرام در ایران فیلتر است؛ ادمین باید پروکسی سرور را تنظیم کند."
            raise PlatformError("ارتباط با اینستاگرام برقرار نشد." + hint, "network", SETTINGS_HELP, str(exc)) from exc
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            raise PlatformError("اینستاگرام پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])
        if isinstance(data, dict) and ("error" in data or "error_type" in data or resp.status_code >= 400):
            raise self._humanize(data)
        return data

    async def _graph(self, app: dict[str, Any], method: str, path: str, token: str,
                     params: dict | None = None) -> dict[str, Any]:
        params = {**(params or {}), "access_token": token}
        url = f"{GRAPH}/{settings.instagram_graph_version}/{path}"
        if method == "POST":
            return await self._http(app, "POST", url, data=params)
        return await self._http(app, "GET", url, params=params)

    @staticmethod
    def _humanize(data: dict[str, Any]) -> PlatformError:
        err = data.get("error") if isinstance(data.get("error"), dict) else data
        code, sub = err.get("code"), err.get("error_subcode")
        msg = err.get("error_user_msg") or err.get("message") or err.get("error_message") or ""
        if code == 190 or err.get("error_type") == "OAuthException" and "token" in msg.lower():
            return PlatformError("دسترسی اینستاگرام منقضی یا لغو شده؛ دوباره وصل کنید.", "expired", RECONNECT, data)
        if code in (10, 200, 3) or sub == 2207050:
            return PlatformError("اپ اجازهٔ انتشار روی این حساب را ندارد؛ حساب باید Business یا Creator باشد و دسترسی انتشار را تأیید کنید.",
                                 "permission", RECONNECT, data)
        if code in (4, 17, 32, 613) or sub in (2207042, 2207051):
            return PlatformError("به سقف انتشار اینستاگرام رسیدید (۱۰۰ پست در ۲۴ ساعت)؛ بعداً دوباره تلاش کنید.",
                                 "rate_limited", {"type": "retry", "label": "تلاش دوباره"}, data)
        if code == 9004 or sub in (2207052, 2207003, 2207026, 2207004, 2207005, 2207009):
            return PlatformError("اینستاگرام نتوانست عکس را دریافت کند؛ عکس باید JPEG و روی آدرس عمومی در دسترس باشد.",
                                 "bad_image", EDIT_PRODUCT, data)
        if sub == 2207028 or "caption" in msg.lower():
            return PlatformError("کپشن برای اینستاگرام مناسب نیست (حداکثر ۲۲۰۰ کاراکتر و ۳۰ هشتگ).",
                                 "caption", EDIT_PRODUCT, data)
        return PlatformError(f"اینستاگرام خطا داد: {msg or 'نامشخص'}", "platform_error", detail=data)

    # ---------- tokens ----------

    async def _long_lived(self, app: dict[str, Any], short_token: str) -> dict[str, Any]:
        return await self._http(app, "GET", f"{GRAPH}/access_token", params={
            "grant_type": "ig_exchange_token", "client_secret": app["app_secret"], "access_token": short_token})

    async def _token(self, ctx, app: dict[str, Any], channel: Channel) -> str:
        creds = decrypt_credentials(channel.credentials_encrypted)
        token = creds.get("access_token", "")
        if not token:
            raise PlatformError("حساب اینستاگرام وصل نیست.", "missing_token", RECONNECT)
        expires = creds.get("expires_at") or 0
        now = time.time()
        if expires and expires - now < REFRESH_BEFORE and now - creds.get("issued_at", 0) > 86400:
            async with ctx.step("refresh_token", "تمدید دسترسی اینستاگرام") as s:
                data = await self._http(app, "GET", f"{GRAPH}/refresh_access_token",
                                        params={"grant_type": "ig_refresh_token", "access_token": token})
                token = data["access_token"]
                channel.credentials_encrypted = encrypt_credentials({
                    **creds, "access_token": token, "issued_at": now, "expires_at": now + int(data.get("expires_in", 0))})
                s.response = {"expires_in": data.get("expires_in")}
        return token

    # ---------- adapter ----------

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        app = await self._app(ctx)
        now = time.time()
        if data.get("code"):
            if not app["oauth_ready"]:
                raise PlatformError("ادمین هنوز شناسه و کلید اپ Meta را وارد نکرده است.", "oauth_not_configured", SETTINGS_HELP)
            async with ctx.step("oauth_code", "دریافت دسترسی از اینستاگرام") as s:
                short = await self._http(app, "POST", OAUTH_TOKEN, data={
                    "client_id": app["app_id"], "client_secret": app["app_secret"], "grant_type": "authorization_code",
                    "redirect_uri": app["redirect_uri"], "code": data["code"].removesuffix("#_")})
                s.response = {"user_id": short.get("user_id"), "permissions": short.get("permissions")}
            async with ctx.step("long_lived", "دریافت توکن بلندمدت (۶۰ روزه)") as s:
                long = await self._long_lived(app, short["access_token"])
                s.response = {"expires_in": long.get("expires_in")}
            token, expires_in = long["access_token"], int(long.get("expires_in") or 0)
        else:
            token, expires_in = (data.get("access_token") or "").strip(), 0
            if not token:
                raise PlatformError("توکن اینستاگرام را وارد کنید.", "missing_token")
        async with ctx.step("account", "بررسی حساب حرفه‌ای") as s:
            me = await self._graph(app, "GET", "me", token, {"fields": "user_id,username,account_type,name"})
            s.response = me
        if me.get("account_type") not in (None, "BUSINESS", "MEDIA_CREATOR", "CREATOR"):
            raise PlatformError("حساب اینستاگرام باید Business یا Creator باشد (از تنظیمات اینستاگرام تغییرش دهید).",
                                "not_professional", RECONNECT, me)
        ig_id = str(me.get("user_id") or me.get("id"))
        return {"credentials": {"access_token": token, "issued_at": now, "expires_at": now + expires_in if expires_in else 0},
                "config": {"ig_user_id": ig_id, "username": me.get("username"), "account_type": me.get("account_type")},
                "display_name": "@" + (me.get("username") or ig_id), "external_account_id": ig_id}

    async def health(self, ctx, channel: Channel) -> None:
        app = await self._app(ctx)
        token = await self._token(ctx, app, channel)
        async with ctx.step("quota", "بررسی دسترسی و سهمیهٔ انتشار") as s:
            s.response = await self._graph(app, "GET", f"{channel.config['ig_user_id']}/content_publishing_limit", token,
                                           {"fields": "quota_usage,config"})

    def validate(self, product: Product) -> list[str]:
        errors = []
        if not product.images:
            errors.append("اینستاگرام حداقل یک عکس لازم دارد.")
        if settings.public_base_url.startswith(("http://localhost", "http://127.")):
            errors.append("اینستاگرام عکس را از آدرس عمومی سرور می‌خواند؛ SELLO_PUBLIC_BASE_URL باید دامنهٔ عمومی باشد.")
        return errors

    async def _wait_ready(self, app, token: str, container: str) -> None:
        for _ in range(20):
            st = await self._graph(app, "GET", container, token, {"fields": "status_code,status"})
            code = st.get("status_code")
            if code in ("FINISHED", "PUBLISHED"):
                return
            if code in ("ERROR", "EXPIRED"):
                raise PlatformError("اینستاگرام پردازش رسانه را رد کرد.", "media_error", EDIT_PRODUCT, st)
            await asyncio.sleep(self.poll_seconds)
        raise PlatformError("پردازش رسانه در اینستاگرام طول کشید؛ کمی بعد دوباره تلاش کنید.", "media_timeout",
                            {"type": "retry", "label": "تلاش دوباره"})

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        app = await self._app(ctx)
        token = await self._token(ctx, app, channel)
        ig = channel.config["ig_user_id"]
        caption = caption_for(product, store.settings or {}, show_price=(getattr(getattr(ctx, "op", None), "payload", None) or {}).get("show_price"))[:CAPTION_LIMIT]
        if not product.images:
            raise PlatformError("اینستاگرام حداقل یک عکس لازم دارد.", "no_image", EDIT_PRODUCT)

        async with ctx.step("quota", "بررسی سهمیهٔ انتشار ۲۴ ساعته") as s:
            q = await self._graph(app, "GET", f"{ig}/content_publishing_limit", token, {"fields": "quota_usage,config"})
            row = (q.get("data") or [{}])[0]
            s.response = row
            if row.get("quota_usage", 0) >= (row.get("config") or {}).get("quota_total", 100):
                raise PlatformError("به سقف ۱۰۰ پست در ۲۴ ساعت رسیدید؛ پست در صف می‌ماند و بعداً منتشر کنید.",
                                    "rate_limited", {"type": "retry", "label": "تلاش دوباره"}, row)

        async with ctx.step("images", "آماده‌سازی عکس‌ها (JPEG روی آدرس عمومی)") as s:
            urls = [await public_jpeg(img.url) for img in product.images[:MAX_CAROUSEL]]
            s.response = {"images": urls}

        async with ctx.step("container", "ساخت کانتینر پست" if len(urls) == 1 else f"ساخت کاروسل {len(urls)} عکسی") as s:
            if len(urls) == 1:
                container = (await self._graph(app, "POST", f"{ig}/media", token,
                                               {"image_url": urls[0], "caption": caption, "alt_text": product.title[:100]}))["id"]
            else:
                children = []
                for u in urls:
                    children.append((await self._graph(app, "POST", f"{ig}/media", token,
                                                       {"image_url": u, "is_carousel_item": "true"}))["id"])
                for c in children:
                    await self._wait_ready(app, token, c)
                container = (await self._graph(app, "POST", f"{ig}/media", token, {
                    "media_type": "CAROUSEL", "children": ",".join(children), "caption": caption}))["id"]
            await self._wait_ready(app, token, container)
            s.response = {"container": container}

        async with ctx.step("publish", "انتشار پست") as s:
            res = await self._graph(app, "POST", f"{ig}/media_publish", token, {"creation_id": container})
            s.response = res
        try:
            permalink = (await self._graph(app, "GET", res["id"], token, {"fields": "permalink"})).get("permalink", "")
        except PlatformError:
            permalink = ""
        return PublishResult(res["id"], permalink)

    async def fetch_items(self, ctx, channel: Channel, limit: int = 50) -> list[CrawledItem]:
        app = await self._app(ctx)
        token = await self._token(ctx, app, channel)
        async with ctx.step("media", "دریافت پست‌های حساب") as s:
            data = await self._graph(app, "GET", f"{channel.config['ig_user_id']}/media", token,
                                     {"fields": "id,caption,media_type,media_url,permalink,children{media_url}",
                                      "limit": min(limit, 100)})
            s.response = {"count": len(data.get("data", []))}
        out = []
        for m in data.get("data", []):
            imgs = [c["media_url"] for c in (m.get("children") or {}).get("data", []) if c.get("media_url")] or (
                [m["media_url"]] if m.get("media_url") and m.get("media_type") != "VIDEO" else [])
            out.append(CrawledItem(external_id=m["id"], caption=m.get("caption", ""), image_urls=imgs,
                                   url=m.get("permalink", ""), raw=m))
        return out


instagram = InstagramAdapter()
