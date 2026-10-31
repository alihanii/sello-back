"""Creates the single Sello plan (1/3/12-month prices) on a fresh install; the admin edits it."""

from sqlalchemy import func, select

from app.core.db import SessionLocal
from app.models import Plan
from app.services.plans import PLAN_TIER

FEATURES = ["۳ فروشگاه", "۸ کانال (تلگرام، بله، دیجی‌کالا، اینستاگرام)", "۲٬۰۰۰ محصول", "۳٬۰۰۰ انتشار در ماه",
            "کرال ساعتی", "۵۰۰ اعتبار AI در ماه", "ساخت محصول با AI"]
LIMITS = {"max_stores": 3, "max_channels": 8, "max_products": 2000, "monthly_publishes": 3000,
          "monthly_ai_credits": 500, "crawl_frequency": "hourly"}
MONTHLY = 990_000
DURATIONS = [(1, 0), (3, 10), (12, 25)]  # months, discount %


async def seed() -> None:
    async with SessionLocal() as db:
        if await db.scalar(select(func.count()).select_from(Plan)):
            return
        for months, discount in DURATIONS:
            full = MONTHLY * months
            db.add(Plan(name="Sello", tier=PLAN_TIER, duration_months=months,
                        price_toman=int(round(full * (100 - discount) / 100, -4)), compare_at_toman=full if discount else 0,
                        description="همهٔ امکانات Sello؛ فقط مدت اشتراک فرق می‌کند.", features=FEATURES, limits=LIMITS,
                        is_featured=months == 12))
        await db.commit()
