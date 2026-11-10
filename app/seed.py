"""Sets up the default plan (1/3/12-month prices) once; the admin edits it, and can add plans, afterwards.

On a database that already has the default plan, its prices, limits and features are moved to the current
defaults once (subscriptions already bought keep their dates)."""

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import AppSetting, Plan
from app.services.plans import DEFAULT_TIERS, tier_plans

SEEDED_KEY = "plans:v2_seeded"


async def seed() -> None:
    async with SessionLocal() as db:
        if await db.get(AppSetting, SEEDED_KEY):
            return
        for t in DEFAULT_TIERS:
            current = {p.duration_months: p for p in (await db.execute(select(Plan).where(Plan.tier == t["tier"]))).scalars()}
            for new in tier_plans(t):
                old = current.get(new.duration_months)
                if not old:
                    db.add(new)
                    continue
                for k in ("name", "price_toman", "compare_at_toman", "description", "features", "limits", "is_featured",
                          "sort_order"):
                    setattr(old, k, getattr(new, k))
        db.add(AppSetting(key=SEEDED_KEY, value={"tiers": [t["tier"] for t in DEFAULT_TIERS]}))
        await db.commit()
