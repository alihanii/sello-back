"""In-app notifications for the store owner (approvals waiting, failed publishes, broken channels)."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Notification, Store


async def notify(db: AsyncSession, store_id: int, kind: str, title_fa: str, body_fa: str = "", link: str = "",
                 image_url: str = "") -> None:
    store = await db.get(Store, store_id)
    if store:
        db.add(Notification(user_id=store.owner_id, store_id=store_id, kind=kind, title_fa=title_fa[:200],
                            body_fa=body_fa, link=link, image_url=image_url))
