from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import utcnow
from app.models import Channel, Plan, Product, Store, Subscription, User

# Limits for users without a subscription (trial / free).
FREE_LIMITS = {"max_stores": 1, "max_channels": 2, "max_products": 30, "monthly_ai_credits": 20,
               "monthly_publishes": 50, "crawl_frequency": "daily"}


async def active_subscription(db: AsyncSession, user_id: int) -> Subscription | None:
    now = utcnow()
    q = (select(Subscription).where(Subscription.user_id == user_id, Subscription.status == "active",
                                    Subscription.starts_at <= now, Subscription.ends_at > now)
         .order_by(Subscription.ends_at.desc()))
    return (await db.execute(q)).scalars().first()


async def limits_for(db: AsyncSession, user: User) -> dict:
    if user.role == "admin":
        return {k: (10**9 if isinstance(v, int) else "hourly") for k, v in FREE_LIMITS.items()}
    sub = await active_subscription(db, user.id)
    if not sub:
        return dict(FREE_LIMITS)
    plan = await db.get(Plan, sub.plan_id)
    return {**FREE_LIMITS, **(plan.limits if plan else {})}


def _limit_error(what: str, limit: int) -> HTTPException:
    return HTTPException(402, {"message": f"به سقف {what} پلن فعلی ({limit}) رسیده‌اید؛ برای ادامه پلن را ارتقا دهید.",
                               "action": {"type": "upgrade", "label": "ارتقای پلن"}})


async def check_limit(db: AsyncSession, user: User, kind: str) -> None:
    limits = await limits_for(db, user)
    if kind == "stores":
        n = await db.scalar(select(func.count()).select_from(Store).where(Store.owner_id == user.id, ~Store.archived))
        if n >= limits["max_stores"]:
            raise _limit_error("تعداد فروشگاه", limits["max_stores"])
    elif kind == "channels":
        n = await db.scalar(select(func.count()).select_from(Channel).join(Store).where(
            Store.owner_id == user.id, Channel.status != "disconnected"))
        if n >= limits["max_channels"]:
            raise _limit_error("تعداد کانال", limits["max_channels"])
    elif kind == "products":
        n = await db.scalar(select(func.count()).select_from(Product).join(Store).where(
            Store.owner_id == user.id, Product.status != "archived"))
        if n >= limits["max_products"]:
            raise _limit_error("تعداد محصول", limits["max_products"])
