"""Login codes by SMS through Kavenegar (https://kavenegar.com).

Settings come from the admin panel (app setting `sms`, API key encrypted) with SELLO_SMS_* / SELLO_KAVENEGAR_* env
fallbacks. Codes go through Verify Lookup with a template approved in the Kavenegar console (fast, allowed on numbers
that block advertising); without a template, a plain sms/send from the configured sender line is used.
"""

import logging
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import AppSetting

log = logging.getLogger("sello.sms")

SMS = "sms"
API = "https://api.kavenegar.com/v1"

# Kavenegar's return.status codes, for messages the admin can act on.
ERRORS = {
    400: "پارامترها ناقص است.",
    401: "حساب کاوه‌نگار غیرفعال است.",
    402: "ارسال انجام نشد.",
    403: "کلید API کاوه‌نگار نامعتبر است.",
    407: "دسترسی به این متد نیست (محدودیت IP در پنل کاوه‌نگار را بررسی کنید).",
    411: "شمارهٔ گیرنده نامعتبر است.",
    412: "خط ارسال‌کننده نامعتبر است.",
    414: "تعداد درخواست بیش از حد مجاز است.",
    418: "اعتبار حساب کاوه‌نگار کافی نیست.",
    422: "متن پیام نویسهٔ غیرمجاز دارد.",
    424: "الگو (template) پیدا نشد یا هنوز تأیید نشده است.",
    426: "استفاده از این متد نیازمند سرویس پیشرفته است.",
    428: "ارسال کد با تماس تلفنی ممکن نیست.",
    431: "ساختار کد صحیح نیست.",
    432: "پارامتر %token در متن الگو نیست.",
}


class SmsError(Exception):
    pass


async def sms_config(db: AsyncSession) -> dict[str, Any]:
    row = await db.get(AppSetting, SMS)
    value = row.value if row else {}
    key = decrypt_credentials(row.secret_encrypted).get("api_key", "") if row and row.secret_encrypted else ""
    provider = value.get("provider") or settings.sms_provider
    enabled = bool(value["enabled"]) if "enabled" in value else bool(settings.sms_provider and settings.sms_api_key)
    cfg = {
        "enabled": enabled,
        "provider": provider or "kavenegar",
        "api_key": key or settings.sms_api_key,
        "template": value.get("template", settings.kavenegar_template),
        "sender": value.get("sender", settings.kavenegar_sender),
    }
    cfg["ready"] = bool(cfg["enabled"] and cfg["api_key"] and (cfg["template"] or cfg["sender"]))
    return cfg


async def save_sms_config(db: AsyncSession, *, enabled: bool, template: str, sender: str, api_key: str | None) -> None:
    row = await db.get(AppSetting, SMS)
    if not row:
        row = AppSetting(key=SMS, value={})
        db.add(row)
    row.value = {"enabled": enabled, "provider": "kavenegar", "template": template.strip(), "sender": sender.strip()}
    if api_key is not None:
        row.secret_encrypted = encrypt_credentials({"api_key": api_key.strip()}) if api_key.strip() else None


def _check(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception:
        raise SmsError(f"پاسخ نامعتبر از کاوه‌نگار (HTTP {resp.status_code}).")
    status = int((data.get("return") or {}).get("status") or resp.status_code)
    if status != 200:
        raise SmsError(ERRORS.get(status) or (data.get("return") or {}).get("message") or f"خطای کاوه‌نگار ({status}).")
    return data


async def send_code(cfg: dict[str, Any], phone: str, code: str) -> None:
    """Sends a login code; raises SmsError with a reason the admin can read."""
    if not cfg.get("api_key"):
        raise SmsError("کلید API کاوه‌نگار تنظیم نشده است.")
    base = f"{API}/{cfg['api_key']}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if cfg.get("template"):
                resp = await client.post(f"{base}/verify/lookup.json",
                                         data={"receptor": phone, "token": code, "template": cfg["template"]})
            elif cfg.get("sender"):
                resp = await client.post(f"{base}/sms/send.json", data={
                    "receptor": phone, "sender": cfg["sender"],
                    "message": f"کد ورود شما به فروشیفای: {code}\nاین کد را به کسی ندهید."})
            else:
                raise SmsError("نام الگو یا خط ارسال‌کننده تنظیم نشده است.")
    except httpx.HTTPError as e:
        # The URL holds the API key: never log it.
        raise SmsError(f"ارتباط با کاوه‌نگار برقرار نشد ({type(e).__name__}).")
    _check(resp)


async def account_info(cfg: dict[str, Any]) -> dict[str, Any]:
    """Credit and account type, to check the API key from the admin panel."""
    if not cfg.get("api_key"):
        raise SmsError("کلید API کاوه‌نگار تنظیم نشده است.")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API}/{cfg['api_key']}/account/info.json")
    except httpx.HTTPError as e:
        raise SmsError(f"ارتباط با کاوه‌نگار برقرار نشد ({type(e).__name__}).")
    entries = _check(resp).get("entries") or {}
    return {"remaining_credit": entries.get("remaincredit"), "type": entries.get("type"), "expiry": entries.get("expiredate")}
