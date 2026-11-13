from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.security import decode_token
from app.models import AuthSession, Store, User

bearer = HTTPBearer(auto_error=False)
DB = Annotated[AsyncSession, Depends(get_db)]


async def user_for_token(db: AsyncSession, token: str) -> tuple[User, AuthSession]:
    """The user and device session behind an access token; 401 if it expired, the session was signed out (logout,
    password change, token reuse) or the account is off."""
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "نشست شما منقضی شده؛ دوباره وارد شوید.")
    session = await db.get(AuthSession, int(payload["sid"]))
    if not session or session.revoked_at is not None or session.user_id != int(payload["sub"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "نشست شما بسته شده؛ دوباره وارد شوید.")
    user = await db.get(User, session.user_id)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "حساب کاربری غیرفعال است.")
    return user, session


async def current_auth(db: DB, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> tuple[User, AuthSession]:
    if not creds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "ابتدا وارد شوید.")
    return await user_for_token(db, creds.credentials)


async def current_user(auth: tuple[User, AuthSession] = Depends(current_auth)) -> User:
    return auth[0]


async def current_session(auth: tuple[User, AuthSession] = Depends(current_auth)) -> AuthSession:
    return auth[1]


CurrentUser = Annotated[User, Depends(current_user)]
CurrentSession = Annotated[AuthSession, Depends(current_session)]


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
