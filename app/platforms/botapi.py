"""Telegram and Bale through their official Bot APIs (Bale mirrors Telegram's shape).
The seller creates a bot, adds it as admin of the channel, and gives us the token + channel."""

import json
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.core.security import decrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, CrawledItem, PlatformError, PublishResult
from app.services.text import MAX_CAPTION_CHARS, caption_for

MAX_ALBUM = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class BotApiAdapter:
    def __init__(self, platform: str, label_fa: str, base_url: str, file_base_url: str):
        self.platform = platform
        self.label_fa = label_fa
        self.base_url = base_url
        self.file_base_url = file_base_url

    # ---------- http ----------

    def _token(self, channel: Channel) -> str:
        token = decrypt_credentials(channel.credentials_encrypted).get("bot_token", "")
        if not token:
            raise PlatformError("توکن ربات ذخیره نشده است.", "missing_token", RECONNECT)
        return token

    async def _call(self, token: str, method: str, data: dict[str, Any] | None = None,
                    files: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/bot{token}/{method}"
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, data=data, files=files) if files else await client.post(url, json=data or {})
        except httpx.HTTPError as exc:
            raise PlatformError(f"ارتباط با {self.label_fa} برقرار نشد.", "network", detail=str(exc)) from exc
        try:
            body = resp.json()
        except ValueError:
            raise PlatformError(f"{self.label_fa} پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])
        if not body.get("ok"):
            raise self._humanize(body)
        return body.get("result")

    def _humanize(self, body: dict[str, Any]) -> PlatformError:
        desc = str(body.get("description", ""))
        low = desc.lower()
        code = body.get("error_code")
        if code == 401 or "unauthorized" in low:
            return PlatformError("توکن ربات نامعتبر است یا ربات حذف شده.", "invalid_token", RECONNECT, body)
        if "chat not found" in low:
            return PlatformError("کانال پیدا نشد؛ شناسهٔ کانال را بررسی کنید و مطمئن شوید ربات عضو آن است.",
                                 "chat_not_found", RECONNECT, body)
        if "not enough rights" in low or "need administrator" in low or "not a member" in low or code == 403:
            return PlatformError("ربات در کانال ادمین نیست یا اجازهٔ ارسال پست ندارد.", "bot_not_admin",
                                 {"type": "help", "label": "راهنمای ادمین کردن ربات"}, body)
        if "caption is too long" in low or "message is too long" in low:
            return PlatformError("متن پست از حد مجاز طولانی‌تر است.", "caption_too_long", EDIT_PRODUCT, body)
        if "wrong file" in low or "failed to get http url content" in low or "photo_invalid" in low:
            return PlatformError("یکی از عکس‌ها قابل ارسال نبود (فرمت یا حجم نامعتبر).", "bad_image", EDIT_PRODUCT, body)
        if code == 429:
            return PlatformError(f"{self.label_fa} موقتاً محدودیت ارسال گذاشته؛ کمی بعد دوباره تلاش کنید.",
                                 "rate_limited", {"type": "retry", "label": "تلاش دوباره"}, body)
        return PlatformError(f"{self.label_fa} خطا داد: {desc or 'نامشخص'}", "platform_error", detail=body)

    # ---------- adapter ----------

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        token = data["bot_token"].strip()
        chat = str(data["chat_id"]).strip()
        if chat and not chat.startswith(("@", "-")) and not chat.lstrip("-").isdigit():
            chat = "@" + chat
        async with ctx.step("get_me", "بررسی توکن ربات", request={"method": "getMe"}) as s:
            me = await self._call(token, "getMe")
            s.response = me
        async with ctx.step("get_chat", "پیدا کردن کانال", request={"method": "getChat", "chat_id": chat}) as s:
            info = await self._call(token, "getChat", {"chat_id": chat})
            s.response = info
        async with ctx.step("check_admin", "بررسی دسترسی ادمین ربات",
                            request={"method": "getChatMember", "chat_id": chat, "user_id": me["id"]}) as s:
            member = await self._call(token, "getChatMember", {"chat_id": chat, "user_id": me["id"]})
            s.response = member
            if member.get("status") not in ("administrator", "creator"):
                raise PlatformError("ربات در کانال ادمین نیست؛ ربات را ادمین کانال کنید و دوباره وصل شوید.",
                                    "bot_not_admin", {"type": "help", "label": "راهنمای ادمین کردن ربات"})
        return {
            "credentials": {"bot_token": token},
            "config": {"chat_id": info.get("id", chat), "username": info.get("username", ""),
                       "bot_username": me.get("username", "")},
            "display_name": info.get("title") or chat,
            "external_account_id": str(info.get("id", chat)),
        }

    async def health(self, ctx, channel: Channel) -> None:
        token = self._token(channel)
        async with ctx.step("check_admin", "بررسی دسترسی ربات در کانال") as s:
            me = await self._call(token, "getMe")
            member = await self._call(token, "getChatMember", {"chat_id": channel.config.get("chat_id"), "user_id": me["id"]})
            s.response = member
            if member.get("status") not in ("administrator", "creator"):
                raise PlatformError("ربات دیگر ادمین کانال نیست.", "bot_not_admin", RECONNECT)

    def validate(self, product: Product) -> list[str]:
        errors = []
        if not product.title.strip():
            errors.append("عنوان محصول خالی است.")
        if len(product.title) > MAX_CAPTION_CHARS - 50:
            errors.append("عنوان محصول خیلی طولانی است.")
        return errors

    async def _image_bytes(self, url: str) -> tuple[bytes, str]:
        if url.startswith("/media/"):
            path = Path(settings.media_dir) / url.removeprefix("/media/")
            if not path.is_file():
                raise PlatformError("فایل عکس محصول پیدا نشد.", "image_missing", EDIT_PRODUCT)
            data = path.read_bytes()
        else:
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.content
            except httpx.HTTPError as exc:
                raise PlatformError("دانلود عکس محصول ناموفق بود.", "image_download", EDIT_PRODUCT, str(exc)) from exc
        if len(data) > MAX_IMAGE_BYTES:
            raise PlatformError("حجم یکی از عکس‌ها بیش از ۱۰ مگابایت است.", "image_too_large", EDIT_PRODUCT)
        name = url.rsplit("/", 1)[-1].split("?")[0] or "photo.jpg"
        return data, name

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        token = self._token(channel)
        chat_id = channel.config.get("chat_id")
        st = store.settings or {}
        async with ctx.step("build_caption", "آماده‌سازی متن پست") as s:
            caption = caption_for(product, st, show_price=(getattr(getattr(ctx, "op", None), "payload", None) or {}).get("show_price"))
            s.response = {"caption": caption, "length": len(caption)}

        images = [img.url for img in product.images][:MAX_ALBUM]
        files: list[tuple[bytes, str]] = []
        if images:
            async with ctx.step("load_images", f"آماده‌سازی {len(images)} عکس", request={"images": images}) as s:
                for url in images:
                    files.append(await self._image_bytes(url))
                s.response = {"loaded": len(files)}

        if existing_remote_id and not files:
            async with ctx.step("edit", "به‌روزرسانی پست قبلی",
                                request={"method": "editMessageText", "message_id": existing_remote_id}) as s:
                s.response = await self._call(token, "editMessageText",
                                              {"chat_id": chat_id, "message_id": int(existing_remote_id), "text": caption})
            return PublishResult(existing_remote_id, self._post_url(channel, existing_remote_id))

        if len(files) > 1:
            media = [{"type": "photo", "media": f"attach://p{i}", **({"caption": caption} if i == 0 else {})}
                     for i in range(len(files))]
            async with ctx.step("send", "ارسال آلبوم به کانال",
                                request={"method": "sendMediaGroup", "chat_id": chat_id, "media": media}) as s:
                result = await self._call(token, "sendMediaGroup", {"chat_id": str(chat_id), "media": json.dumps(media)},
                                          files={f"p{i}": (n, d) for i, (d, n) in enumerate(files)})
                s.response = result
                message_id = str(result[0]["message_id"])
        elif files:
            async with ctx.step("send", "ارسال عکس و متن به کانال",
                                request={"method": "sendPhoto", "chat_id": chat_id, "caption": caption}) as s:
                d, n = files[0]
                result = await self._call(token, "sendPhoto", {"chat_id": str(chat_id), "caption": caption},
                                          files={"photo": (n, d)})
                s.response = result
                message_id = str(result["message_id"])
        else:
            async with ctx.step("send", "ارسال متن به کانال",
                                request={"method": "sendMessage", "chat_id": chat_id, "text": caption}) as s:
                result = await self._call(token, "sendMessage", {"chat_id": chat_id, "text": caption})
                s.response = result
                message_id = str(result["message_id"])
        return PublishResult(message_id, self._post_url(channel, message_id))

    def _post_url(self, channel: Channel, message_id: str) -> str:
        username = channel.config.get("username")
        host = "t.me" if self.platform == "telegram" else "ble.ir"
        return f"https://{host}/{username}/{message_id}" if username else ""

    async def fetch_items(self, ctx, channel: Channel, limit: int = 50) -> list[CrawledItem]:
        """Bots can't read channel history; we read channel posts the bot has received
        since it was added (getUpdates), newest last."""
        token = self._token(channel)
        offset = int(channel.config.get("update_offset", 0))
        async with ctx.step("get_updates", "دریافت پست‌های جدید کانال",
                            request={"method": "getUpdates", "offset": offset}) as s:
            updates = await self._call(token, "getUpdates",
                                       {"offset": offset, "limit": 100, "allowed_updates": ["channel_post", "edited_channel_post"]})
            s.response = {"count": len(updates)}
        items: dict[str, CrawledItem] = {}
        groups: dict[str, CrawledItem] = {}
        chat_id = str(channel.config.get("chat_id"))
        other_chats: set[str] = set()
        for upd in updates:
            offset = max(offset, upd["update_id"] + 1)
            post = upd.get("channel_post") or upd.get("edited_channel_post")
            if not post:
                continue
            if str(post.get("chat", {}).get("id")) != chat_id:
                other_chats.add(post.get("chat", {}).get("title") or str(post.get("chat", {}).get("id")))
                continue
            photo = (post.get("photo") or [{}])[-1].get("file_id")
            caption = post.get("caption") or post.get("text") or ""
            key = post.get("media_group_id") or str(post["message_id"])
            item = groups.get(key) if post.get("media_group_id") else None
            if item is None:
                item = CrawledItem(external_id=str(post["message_id"]), caption=caption, raw=post,
                                   url=self._post_url(channel, str(post["message_id"])))
                if post.get("media_group_id"):
                    groups[key] = item
                items[item.external_id] = item
            elif caption and not item.caption:
                item.caption = caption
            if photo:
                item.image_urls.append(f"botfile:{photo}")
        channel.config = {**channel.config, "update_offset": offset}
        if not items:
            ctx.empty_reason = self._empty_reason(bool(channel.config.get("crawled_once")), other_chats)
        channel.config = {**channel.config, "crawled_once": True}
        return list(items.values())[-limit:]

    def _empty_reason(self, crawled_before: bool, other_chats: set[str]) -> str:
        name = self.label_fa
        if other_chats:
            return (f"ربات پست‌هایی دریافت کرد ولی از کانال دیگری ({'، '.join(sorted(other_chats))[:120]}) بودند، "
                    "نه کانالی که اینجا وصل است. کانال درست را در صفحهٔ کانال‌ها وصل کن.")
        base = (f"ربات {name} نمی‌تواند پست‌های قدیمی کانال را بخواند؛ فقط پست‌هایی را می‌بیند که بعد از ادمین شدنش "
                "در کانال منتشر شده‌اند، و پیام‌رسان این‌ها را حداکثر ۲۴ ساعت نگه می‌دارد. ")
        if crawled_before:
            return base + "از کرال قبلی تا الان پست جدیدی در کانال منتشر نشده. یک پست محصول بگذار و دوباره کرال کن."
        return base + "یک پست محصول (عکس + متن با قیمت) در کانال بگذار و دوباره کرال کن."


    async def download_file(self, channel: Channel, file_id: str) -> tuple[bytes, str]:
        token = self._token(channel)
        info = await self._call(token, "getFile", {"file_id": file_id})
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(f"{self.file_base_url}/file/bot{token}/{info['file_path']}")
            resp.raise_for_status()
        return resp.content, info["file_path"].rsplit("/", 1)[-1]


telegram = BotApiAdapter("telegram", "تلگرام", "https://api.telegram.org", "https://api.telegram.org")
bale = BotApiAdapter("bale", "بله", "https://tapi.bale.ai", "https://tapi.bale.ai")
