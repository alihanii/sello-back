"""Integration settings the admin edits in the panel, with env fallbacks."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import AppSetting

INSTAGRAM = "instagram"
INSTAGRAM_SCOPES = "instagram_business_basic,instagram_business_content_publish"
# How sellers connect: "login" = username + password through Instagram's app API (instagrapi), "oauth" = Meta's
# official Instagram Login (needs a reviewed Meta app). Pacing applies to "login" accounts, to keep them safe.
IG_METHODS = ("login", "oauth")
IG_PACING = {"daily_limit": 8, "gap_min_minutes": 20, "gap_max_minutes": 45}


async def instagram_app(db: AsyncSession) -> dict[str, Any]:
    row = await db.get(AppSetting, INSTAGRAM)
    value = row.value if row else {}
    secret = decrypt_credentials(row.secret_encrypted).get("app_secret", "") if row else ""
    app_id = value.get("app_id") or settings.instagram_app_id
    app_secret = secret or settings.instagram_app_secret
    return {
        "enabled": bool(value.get("enabled", settings.instagram_enabled)),
        "app_id": app_id,
        "app_secret": app_secret,
        "proxy": value.get("proxy", settings.instagram_proxy),
        "method": value.get("method") if value.get("method") in IG_METHODS else "login",
        **{k: int(value.get(k) or v) for k, v in IG_PACING.items()},
        "oauth_ready": bool(app_id and app_secret),
        "redirect_uri": settings.frontend_base_url.rstrip("/") + "/app/oauth/instagram",
    }


async def save_instagram_app(db: AsyncSession, *, enabled: bool, app_id: str, proxy: str,
                             app_secret: str | None, method: str = "", pacing: dict[str, int] | None = None) -> None:
    row = await db.get(AppSetting, INSTAGRAM)
    if not row:
        row = AppSetting(key=INSTAGRAM, value={})
        db.add(row)
    old = row.value or {}
    row.value = {"enabled": enabled, "app_id": app_id.strip(), "proxy": proxy.strip(),
                 "method": method if method in IG_METHODS else old.get("method", "login"),
                 **{k: int((pacing or {}).get(k) or old.get(k) or v) for k, v in IG_PACING.items()}}
    if app_secret is not None:
        row.secret_encrypted = encrypt_credentials({"app_secret": app_secret.strip()}) if app_secret.strip() else None


INSTAGRAM_CRAWLER = "instagram_crawler"


async def instagram_crawler(db: AsyncSession) -> dict[str, Any]:
    """Scraping service for crawling public Instagram pages by username: {provider, token, proxy}."""
    row = await db.get(AppSetting, INSTAGRAM_CRAWLER)
    value = row.value if row else {}
    token = decrypt_credentials(row.secret_encrypted).get("token", "") if row and row.secret_encrypted else ""
    return {"provider": value.get("provider") or "apify", "token": token, "proxy": value.get("proxy", "")}


async def save_instagram_crawler(db: AsyncSession, *, provider: str, proxy: str, token: str | None) -> None:
    row = await db.get(AppSetting, INSTAGRAM_CRAWLER)
    if not row:
        row = AppSetting(key=INSTAGRAM_CRAWLER, value={})
        db.add(row)
    row.value = {"provider": provider, "proxy": proxy.strip()}
    if token is not None:
        row.secret_encrypted = encrypt_credentials({"token": token.strip()}) if token.strip() else None


PAYMENT = "payment"


async def payment_config(db: AsyncSession) -> dict[str, Any]:
    """{enabled, bank, gateways: {BANK: {FIELD: value}}}; gateway credentials are stored encrypted."""
    row = await db.get(AppSetting, PAYMENT)
    value = row.value if row else {}
    gateways = decrypt_credentials(row.secret_encrypted).get("gateways", {}) if row and row.secret_encrypted else {}
    return {"enabled": bool(value.get("enabled")), "bank": value.get("bank") or "ZARINPAL", "gateways": gateways}


async def save_payment_config(db: AsyncSession, *, enabled: bool, bank: str, gateways: dict[str, dict[str, Any]]) -> None:
    row = await db.get(AppSetting, PAYMENT)
    if not row:
        row = AppSetting(key=PAYMENT, value={})
        db.add(row)
    row.value = {"enabled": enabled, "bank": bank}
    row.secret_encrypted = encrypt_credentials({"gateways": gateways})


LINKEDIN = "linkedin"
# Sellers sign in with their own LinkedIn email + password (or paste the browser's li_at cookie); LinkedIn has no
# public API for comments or messages, so this goes through LinkedIn's own web API. Paced to keep accounts safe.
LI_PACING = {"daily_limit": 3, "gap_min_minutes": 90, "gap_max_minutes": 180}


async def linkedin_app(db: AsyncSession) -> dict[str, Any]:
    """{enabled, daily_limit, gap_min_minutes, gap_max_minutes, query_ids}. `query_ids` lets the admin replace
    LinkedIn's messaging GraphQL ids from the panel when LinkedIn changes them, without a deploy."""
    row = await db.get(AppSetting, LINKEDIN)
    value = row.value if row else {}
    return {"enabled": bool(value.get("enabled")),
            **{k: int(value.get(k) or v) for k, v in LI_PACING.items()},
            "query_ids": {k: str(v) for k, v in (value.get("query_ids") or {}).items() if v}}


async def save_linkedin_app(db: AsyncSession, *, enabled: bool, pacing: dict[str, int],
                            query_ids: dict[str, str] | None = None) -> None:
    row = await db.get(AppSetting, LINKEDIN)
    if not row:
        row = AppSetting(key=LINKEDIN, value={})
        db.add(row)
    old = row.value or {}
    row.value = {"enabled": enabled, **{k: int(pacing.get(k) or old.get(k) or v) for k, v in LI_PACING.items()},
                 "query_ids": {k: v.strip() for k, v in (query_ids if query_ids is not None else old.get("query_ids") or {}).items()
                               if v and v.strip()}}


DIGIKALA = "digikala"
# Orders are read from the seller panel on a timer (Digikala has no webhooks for sellers).
DK_SYNC_MINUTES = 15


async def digikala_app(db: AsyncSession) -> dict[str, Any]:
    """{enabled, sync_minutes}. Off: sellers can't connect, publish or sync orders; connected channels stay as they are."""
    row = await db.get(AppSetting, DIGIKALA)
    value = row.value if row else {}
    return {"enabled": bool(value.get("enabled", True)),
            "sync_minutes": max(5, int(value.get("sync_minutes") or DK_SYNC_MINUTES))}


async def save_digikala_app(db: AsyncSession, *, enabled: bool, sync_minutes: int | None = None) -> None:
    row = await db.get(AppSetting, DIGIKALA)
    if not row:
        row = AppSetting(key=DIGIKALA, value={})
        db.add(row)
    old = row.value or {}
    row.value = {**old, "enabled": enabled, "sync_minutes": int(sync_minutes or old.get("sync_minutes") or DK_SYNC_MINUTES)}
