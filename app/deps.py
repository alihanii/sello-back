from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.security import decode_token
from app.models import Store, User

bearer = HTTPBearer(auto_error=False)
DB = Annotated[AsyncSession, Depends(get_db)]


async def current_user(db: DB, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
    if not creds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "ابتدا وارد شوید.")
    try:
        payload = decode_token(creds.credentials)
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "نشست شما منقضی شده؛ دوباره وارد شوید.")
    user = await db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "حساب کاربری غیرفعال است.")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def admin_user(user: CurrentUser) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "دسترسی فقط برای ادمین.")
    return user


AdminUser = Annotated[User, Depends(admin_user)]


async def get_store_for(db: AsyncSession, user: User, store_id: int) -> Store:
    store = await db.get(Store, store_id)
    if not store or (store.owner_id != user.id and user.role != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "فروشگاه پیدا نشد.")
    return store
