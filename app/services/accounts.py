"""Account helpers shared by the admin panel and the command line."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User


async def make_admin(db: AsyncSession, phone: str) -> tuple[User, bool]:
    """Admin account for a normalized phone: creates it, or promotes (and re-enables) an existing one.
    Returns (user, created). The caller commits."""
    user = (await db.execute(select(User).where(User.phone == phone))).scalar_one_or_none()
    if user:
        user.role, user.is_active = "admin", True
        return user, False
    user = User(phone=phone, role="admin", is_active=True)
    db.add(user)
    await db.flush()
    return user, True
