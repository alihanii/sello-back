"""Digikala seller panel. HTTP calls ported from platform-wrappers
(digikala.api/app/services/digikala_seller_client.py and digikala_seller_products.py).
These are the seller PWA's internal endpoints, not a public API, so they can change."""

import re
from typing import Any

import httpx

from app.core.security import decrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import RECONNECT, CrawledItem, PlatformError, PublishResult

BASE_URL = "https://seller.digikala.com"
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "x-web-optimize-response": "1",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/pwa/account/sign-in",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0 Safari/537.36",
}


def _digits(v: str) -> str:
    return re.sub(r"\D", "", v or "")


def login_username(phone: str) -> str:
    d = _digits(phone)
    return "0" + d[2:] if d.startswith("98") and len(d) == 12 else d


def verify_username(phone: str) -> str:
    d = _digits(phone)
    return d if d.startswith("98") else "98" + d.lstrip("0")


class DigikalaAdapter:
    platform = "digikala"
    label_fa = "دیجی‌کالا"

    async def _request(self, method: str, path: str, *, token: str | None = None, body: dict | None = None,
                       referer: str | None = None) -> dict[str, Any]:
        headers = dict(DEFAULT_HEADERS)
        if referer:
            headers["Referer"] = referer
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
                resp = await client.request(method, f"{BASE_URL}{path}", json=body)
        except httpx.HTTPError as exc:
            raise PlatformError("ارتباط با پنل فروشندگان دیجی‌کالا برقرار نشد.", "network", detail=str(exc)) from exc
        if resp.status_code == 401:
            raise PlatformError("نشست دیجی‌کالا منقضی شده؛ دوباره با کد تأیید وارد شوید.", "expired", RECONNECT,
                                resp.text[:500])
        if resp.status_code >= 400:
            msg = self._extract_message(resp)
            raise PlatformError(f"دیجی‌کالا خطا داد: {msg}" if msg else "دیجی‌کالا درخواست را رد کرد.",
                                f"http_{resp.status_code}", detail=resp.text[:1000])
        try:
            return resp.json()
        except ValueError:
            raise PlatformError("دیجی‌کالا پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])

    @staticmethod
    def _extract_message(resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except ValueError:
            return ""
        for key in ("message", "error", "errors"):
            val = data.get(key) if isinstance(data, dict) else None
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return " ".join(str(v) for v in val.values())[:300]
        return ""

    # ---------- connect: two calls (send code, then verify) ----------

    async def send_code(self, ctx, phone: str) -> dict[str, Any]:
        username = login_username(phone)
        async with ctx.step("login", "شروع ورود به پنل فروشنده", request={"username": username}) as s:
            s.response = await self._request("POST", "/api/v2/login", body={"username": username, "captcha_token": ""})
        async with ctx.step("send_otp", "ارسال کد تأیید دیجی‌کالا", request={"username": username}) as s:
            payload = await self._request("POST", "/api/v2/otp/send", body={"username": username, "captcha_token": ""})
            s.response = payload
        challenge = (payload.get("data") or {}).get("challenge_token")
        if not challenge:
            raise PlatformError("دیجی‌کالا کد تأیید را ارسال نکرد؛ چند دقیقه بعد دوباره تلاش کنید.", "no_challenge")
        return {"challenge_token": challenge, "phone": username}

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        pending = channel.config.get("pending") or {}
        if not pending.get("challenge_token"):
            raise PlatformError("اول کد تأیید را درخواست کنید.", "no_pending_login")
        async with ctx.step("verify_otp", "تأیید کد", request={"username": verify_username(pending["phone"])}) as s:
            payload = await self._request(
                "POST", "/api/v2/otp/verify", referer=f"{BASE_URL}/pwa/account/otp",
                body={"otp": data["code"], "username": verify_username(pending["phone"]),
                      "challenge_token": pending["challenge_token"]})
            s.response = {"ok": True}
        token = ((payload.get("data") or {}).get("token") or {})
        if not token.get("access_token"):
            raise PlatformError("کد تأیید اشتباه یا منقضی است.", "bad_otp", RECONNECT)
        async with ctx.step("profile", "دریافت مشخصات فروشنده") as s:
            profile = (await self._request("GET", "/api/v2/profile?mini=true", token=token["access_token"],
                                           referer=f"{BASE_URL}/pwa/profile")).get("data") or {}
            s.response = {k: profile.get(k) for k in ("seller_id", "business_name", "name") if k in profile}
        return {
            "credentials": {"access_token": token["access_token"], "refresh_token": token.get("refresh_token")},
            "config": {"phone": pending["phone"], "seller_id": profile.get("seller_id"), "pending": None},
            "display_name": profile.get("business_name") or profile.get("name") or f"دیجی‌کالا {pending['phone']}",
            "external_account_id": str(profile.get("seller_id") or pending["phone"]),
        }

    def _token(self, channel: Channel) -> str:
        token = decrypt_credentials(channel.credentials_encrypted).get("access_token")
        if not token:
            raise PlatformError("حساب دیجی‌کالا وصل نیست.", "missing_token", RECONNECT)
        return token

    async def health(self, ctx, channel: Channel) -> None:
        async with ctx.step("profile", "بررسی نشست دیجی‌کالا") as s:
            data = await self._request("GET", "/api/v2/profile?mini=true", token=self._token(channel),
                                       referer=f"{BASE_URL}/pwa/profile")
            s.response = {"seller_id": (data.get("data") or {}).get("seller_id")}

    def validate(self, product: Product) -> list[str]:
        errors = []
        if len(product.description or "") < 150:
            errors.append("دیجی‌کالا توضیح حداقل ۱۵۰ کاراکتری لازم دارد.")
        if not product.images:
            errors.append("دیجی‌کالا حداقل یک عکس لازم دارد.")
        return errors

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        async with ctx.step("pipeline", "پایپ‌لاین ساخت محصول دیجی‌کالا") as s:
            s.response = {"steps": ["category", "draft", "attributes", "title", "images"]}
            raise PlatformError(
                "انتشار خودکار در دیجی‌کالا در نسخهٔ بعدی فعال می‌شود (در حال انتقال از platform-wrappers).",
                "not_available_yet", {"type": "info", "label": "فعلاً فقط کرال محصولات دیجی‌کالا فعال است"})

    async def fetch_items(self, ctx, channel: Channel, limit: int = 200) -> list[CrawledItem]:
        token = self._token(channel)
        items: list[CrawledItem] = []
        page = 1
        while len(items) < limit:
            async with ctx.step(f"page_{page}", f"دریافت صفحهٔ {page} محصولات فروشنده",
                                request={"page": page, "size": 50}) as s:
                data = (await self._request("GET", f"/api/v2/products/seller?page={page}&size=50&order=desc&categoryId=",
                                            token=token, referer=f"{BASE_URL}/pwa/product-management")).get("data") or {}
                rows = data.get("items") or []
                s.response = {"count": len(rows), "pager": data.get("pager")}
            for row in rows:
                pid = row.get("product_id")
                cat = row.get("main_category_title")
                items.append(CrawledItem(
                    external_id=f"DKP-{pid}", title=row.get("title_fa") or row.get("title") or "",
                    brand=row.get("brand_title_fa") or "",
                    category=next(iter(cat.values()), "") if isinstance(cat, dict) and cat else "",
                    image_urls=[row["image_src"]] if row.get("image_src") else [],
                    url=row.get("product_url") or "", sku=f"DKP-{pid}", raw=row))
            pager = data.get("pager") or {}
            if not rows or page >= int(pager.get("total_pages") or 1):
                break
            page += 1
        return items[:limit]


digikala = DigikalaAdapter()
