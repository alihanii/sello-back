"""Sample plans so the pricing page isn't empty on a fresh install; the admin edits them."""

from sqlalchemy import func, select

from app.core.db import SessionLocal
from app.models import Plan

TIERS = [
    ("پایه", "basic", 490_000, False, "برای شروع فروش در کانال‌ها",
     ["۱ فروشگاه", "۲ کانال", "۲۰۰ محصول", "۳۰۰ انتشار در ماه", "کرال روزانه", "۵۰ اعتبار AI در ماه"],
     {"max_stores": 1, "max_channels": 2, "max_products": 200, "monthly_publishes": 300, "monthly_ai_credits": 50, "crawl_frequency": "daily"}),
    ("حرفه‌ای", "pro", 1_290_000, True, "برای فروشگاه‌هایی که هر روز محصول می‌گذارند",
     ["۳ فروشگاه", "۸ کانال", "۲٬۰۰۰ محصول", "۳٬۰۰۰ انتشار در ماه", "کرال ساعتی", "۵۰۰ اعتبار AI در ماه", "اتوماسیون انتشار", "تولید محصول از رفرنس"],
     {"max_stores": 3, "max_channels": 8, "max_products": 2000, "monthly_publishes": 3000, "monthly_ai_credits": 500, "crawl_frequency": "hourly"}),
    ("تجاری", "business", 2_990_000, False, "برای برندها و چند فروشگاه",
     ["۱۰ فروشگاه", "کانال نامحدود", "محصول نامحدود", "انتشار نامحدود", "کرال ساعتی", "۳٬۰۰۰ اعتبار AI در ماه", "پشتیبانی اختصاصی"],
     {"max_stores": 10, "max_channels": 1000, "max_products": 1_000_000, "monthly_publishes": 1_000_000, "monthly_ai_credits": 3000, "crawl_frequency": "hourly"}),
]
DURATIONS = [(1, 0), (3, 10), (12, 25)]  # months, discount %


async def seed() -> None:
    async with SessionLocal() as db:
        if await db.scalar(select(func.count()).select_from(Plan)):
            return
        for order, (name, tier, monthly, featured, desc, features, limits) in enumerate(TIERS):
            for months, discount in DURATIONS:
                full = monthly * months
                price = round(full * (100 - discount) / 100, -4)
                db.add(Plan(name=name, tier=tier, duration_months=months, price_toman=int(price),
                            compare_at_toman=full if discount else 0, description=desc, features=features,
                            limits=limits, is_featured=featured, sort_order=order))
        await db.commit()
