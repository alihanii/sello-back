"""Instagram via the official Graph API (Business/Creator account linked to a Facebook page).
Behind SELLO_INSTAGRAM_ENABLED. The seller pastes a long-lived token from the Meta app; full
OAuth arrives once the app passes Meta review."""

from typing import Any

import httpx

from app.core.config import settings
from app.core.security import decrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, CrawledItem, PlatformError, PublishResult
from app.services.text import build_caption

GRAPH = "https://graph.facebook.com"


class InstagramAdapter:
    platform = "instagram"
    label_fa = "اینستاگرام"

    def _guard(self) -> None:
        if not settings.instagram_enabled:
            raise PlatformError("اتصال اینستاگرام هنوز فعال نشده است (در انتظار تأیید اپ Meta).", "disabled",
                                {"type": "info", "label": "به‌زودی"})

    async def _call(self, method: str, path: str, token: str, params: dict | None = None) -> dict[str, Any]:
        params = {**(params or {}), "access_token": token}
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.request(method, f"{GRAPH}/{settings.instagram_graph_version}/{path}", params=params)
        except httpx.HTTPError as exc:
            raise PlatformError("ارتباط با اینستاگرام برقرار نشد.", "network", detail=str(exc)) from exc
        data = resp.json() if resp.content else {}
        if "error" in data:
            err = data["error"]
            if err.get("code") == 190:
                raise PlatformError("توکن اینستاگرام منقضی شده.", "expired", RECONNECT, data)
            raise PlatformError(f"اینستاگرام خطا داد: {err.get('message', '')}", "platform_error", detail=data)
        return data

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        self._guard()
        token, ig_id = data["access_token"], data["ig_user_id"]
        async with ctx.step("account", "بررسی حساب بیزینسی", request={"ig_user_id": ig_id}) as s:
            info = await self._call("GET", ig_id, token, {"fields": "username,name"})
            s.response = info
        return {"credentials": {"access_token": token}, "config": {"ig_user_id": ig_id, "username": info.get("username")},
                "display_name": "@" + info.get("username", ig_id), "external_account_id": ig_id}

    async def health(self, ctx, channel: Channel) -> None:
        self._guard()
        token = decrypt_credentials(channel.credentials_encrypted).get("access_token", "")
        async with ctx.step("account", "بررسی توکن اینستاگرام"):
            await self._call("GET", channel.config["ig_user_id"], token, {"fields": "username"})

    def validate(self, product: Product) -> list[str]:
        errors = []
        if not product.images:
            errors.append("اینستاگرام حداقل یک عکس لازم دارد.")
        if any(img.url.startswith("/media/") for img in product.images) and settings.public_base_url.startswith("http://localhost"):
            errors.append("عکس‌ها باید از آدرس عمومی در دسترس باشند.")
        return errors

    def _public(self, url: str) -> str:
        return settings.public_base_url.rstrip("/") + url if url.startswith("/") else url

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        self._guard()
        token = decrypt_credentials(channel.credentials_encrypted).get("access_token", "")
        ig = channel.config["ig_user_id"]
        st = store.settings or {}
        caption = build_caption(title=product.title, description=product.description, price=product.price,
                                show_price=st.get("show_price_in_caption", True), signature=st.get("caption_signature", ""))
        urls = [self._public(i.url) for i in product.images][:10]
        if not urls:
            raise PlatformError("اینستاگرام حداقل یک عکس لازم دارد.", "no_image", EDIT_PRODUCT)
        async with ctx.step("containers", "ساخت کانتینر عکس‌ها", request={"images": urls}) as s:
            if len(urls) == 1:
                container = (await self._call("POST", f"{ig}/media", token, {"image_url": urls[0], "caption": caption}))["id"]
            else:
                children = [(await self._call("POST", f"{ig}/media", token, {"image_url": u, "is_carousel_item": "true"}))["id"]
                            for u in urls]
                container = (await self._call("POST", f"{ig}/media", token,
                                              {"media_type": "CAROUSEL", "children": ",".join(children), "caption": caption}))["id"]
            s.response = {"container": container}
        async with ctx.step("publish", "انتشار پست") as s:
            res = await self._call("POST", f"{ig}/media_publish", token, {"creation_id": container})
            s.response = res
        info = await self._call("GET", res["id"], token, {"fields": "permalink"})
        return PublishResult(res["id"], info.get("permalink", ""))

    async def fetch_items(self, ctx, channel: Channel, limit: int = 50) -> list[CrawledItem]:
        self._guard()
        token = decrypt_credentials(channel.credentials_encrypted).get("access_token", "")
        async with ctx.step("media", "دریافت پست‌های حساب") as s:
            data = await self._call("GET", f"{channel.config['ig_user_id']}/media", token,
                                    {"fields": "id,caption,media_url,permalink,children{media_url}", "limit": limit})
            s.response = {"count": len(data.get("data", []))}
        out = []
        for m in data.get("data", []):
            imgs = [c["media_url"] for c in (m.get("children") or {}).get("data", []) if c.get("media_url")] or (
                [m["media_url"]] if m.get("media_url") else [])
            out.append(CrawledItem(external_id=m["id"], caption=m.get("caption", ""), image_urls=imgs,
                                   url=m.get("permalink", ""), raw=m))
        return out


instagram = InstagramAdapter()
