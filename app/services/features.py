"""Features the admin opens to a segment of sellers (admin panel › قابلیت‌ها).

Each feature is one AppSetting row: {"mode": "all" | "selected" | "none", "user_ids": [...]}. Admins always have it,
so a feature can be tried before anyone else sees it.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting, User

SITE_BUILDER = "site_builder"
FEATURES = {SITE_BUILDER: "وبسایت ساز"}
MODES = ("all", "selected", "none")
# New features start closed: only the users the admin picks get them.
DEFAULT_MODE = "selected"


async def feature_access(db: AsyncSession, key: str) -> dict[str, Any]:
    row = await db.get(AppSetting, f"feature:{key}")
    v = row.value if row else {}
    mode = v.get("mode") if v.get("mode") in MODES else DEFAULT_MODE
    return {"mode": mode, "user_ids": [int(i) for i in v.get("user_ids", [])]}


async def save_feature_access(db: AsyncSession, key: str, mode: str, user_ids: list[int]) -> dict[str, Any]:
    row = await db.get(AppSetting, f"feature:{key}")
    if not row:
        row = AppSetting(key=f"feature:{key}", value={})
        db.add(row)
    row.value = {"mode": mode, "user_ids": sorted(set(int(i) for i in user_ids))}
    return row.value


def allowed(access: dict[str, Any], user: User) -> bool:
    if user.role == "admin":
        return True
    return access["mode"] == "all" or (access["mode"] == "selected" and user.id in access["user_ids"])


async def has_feature(db: AsyncSession, user: User, key: str) -> bool:
    return allowed(await feature_access(db, key), user)


async def user_features(db: AsyncSession, user: User) -> dict[str, bool]:
    return {k: await has_feature(db, user, k) for k in FEATURES}
