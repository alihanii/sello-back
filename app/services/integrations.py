"""Integration settings the admin edits in the panel, with env fallbacks."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decrypt_credentials, encrypt_credentials
from app.models import AppSetting

INSTAGRAM = "instagram"
INSTAGRAM_SCOPES = "instagram_business_basic,instagram_business_content_publish"


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
        "oauth_ready": bool(app_id and app_secret),
        "redirect_uri": settings.frontend_base_url.rstrip("/") + "/app/oauth/instagram",
    }


async def save_instagram_app(db: AsyncSession, *, enabled: bool, app_id: str, proxy: str,
                             app_secret: str | None) -> None:
    row = await db.get(AppSetting, INSTAGRAM)
    if not row:
        row = AppSetting(key=INSTAGRAM, value={})
        db.add(row)
    row.value = {"enabled": enabled, "app_id": app_id.strip(), "proxy": proxy.strip()}
    if app_secret is not None:
        row.secret_encrypted = encrypt_credentials({"app_secret": app_secret.strip()}) if app_secret.strip() else None
