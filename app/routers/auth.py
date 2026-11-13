import logging
import random
import re
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from app.core import ratelimit
from app.core.config import settings
from app.core.db import utcnow
from app.core.passwords import hash_password, password_problem, verify_password
from app.core.security import create_access_token, decode_legacy_token, hash_code, new_refresh_token, refresh_hash
from app.deps import DB, CurrentSession, CurrentUser
from app.models import AuthSession, OtpCode, OtpThrottle, User
from app.schemas import UserOut
from app.services.features import user_features
from app.services.sms import SmsError, send_code, sms_config
from app.services.text import to_en_digits

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("sello.auth")

COOKIE_PATH = "/api/v1/auth"
# Two tabs refreshing at the same moment: the one that loses still gets an access token for this long.
ROTATION_GRACE_SECONDS = 30
MAX_PASSWORD_FAILS = 5
LOCK_MINUTES = 15
# Wrong SMS codes: wait before the next try, and a long lock after this many in a row.
OTP_RETRY_WAIT_SECONDS = 40
OTP_MAX_FAILS = 5
OTP_LOCK_SECONDS = 3600


def _wait(message: str, seconds: int, status: int = 429) -> HTTPException:
    return HTTPException(status, {"message": message, "retry_after": max(1, int(seconds))})


def _lock_message(seconds: float) -> str:
    return f"به‌خاطر چند بار وارد کردن کد اشتباه، ورود با کد پیامکی تا {max(1, round(seconds / 60))} دقیقهٔ دیگر بسته است."


def normalize_phone(phone: str) -> str:
    d = re.sub(r"\D", "", to_en_digits(phone))
    if d.startswith("98"):
        d = "0" + d[2:]
    if not d.startswith("0"):
        d = "0" + d
    if not re.fullmatch(r"09\d{9}", d):
        raise HTTPException(422, "شمارهٔ موبایل معتبر نیست.")
    return d


def aware(dt):
    return dt.replace(tzinfo=utcnow().tzinfo) if dt and dt.tzinfo is None else dt


# ---------- sessions ----------


def _check_origin(request: Request) -> None:
    """Cookie endpoints: refuse calls from other sites (the cookie is also SameSite=Strict)."""
    origin = request.headers.get("origin")
    if not origin:
        return
    allowed = {o.rstrip("/") for o in settings.cors_origins} | {settings.frontend_base_url.rstrip("/"),
                                                                 settings.public_base_url.rstrip("/")}
    if origin.rstrip("/") not in allowed:
        raise HTTPException(403, "درخواست از مبدأ نامعتبر.")


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(settings.refresh_cookie_name, token, max_age=settings.refresh_idle_days * 86400,
                        path=COOKIE_PATH, httponly=True, samesite="strict",
                        secure=settings.public_base_url.startswith("https"))


def _clear_cookie(response: Response) -> None:
    response.delete_cookie(settings.refresh_cookie_name, path=COOKIE_PATH, httponly=True, samesite="strict",
                           secure=settings.public_base_url.startswith("https"))


def _access(user: User, session: AuthSession) -> dict:
    return {"access_token": create_access_token(user.id, user.role, session.id), "token_type": "bearer",
            "expires_in": settings.jwt_access_minutes * 60}


async def start_session(db: DB, request: Request, response: Response, user: User, method: str) -> dict:
    token, digest = new_refresh_token()
    now = utcnow()
    absolute = now + timedelta(days=settings.refresh_absolute_days)
    session = AuthSession(user_id=user.id, token_hash=digest, method=method, last_used_at=now,
                          expires_at=min(now + timedelta(days=settings.refresh_idle_days), absolute),
                          absolute_expires_at=absolute, ip=ratelimit.client_ip(request)[:64],
                          user_agent=(request.headers.get("user-agent") or "")[:300])
    db.add(session)
    user.last_login_at = now
    await db.commit()
    await db.refresh(user)
    _set_cookie(response, token)
    return {**_access(user, session), "user": UserOut.model_validate(user), "has_password": bool(user.password_hash)}


async def revoke_sessions(db: DB, user_id: int, reason: str, keep: int | None = None) -> None:
    q = update(AuthSession).where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
    if keep:
        q = q.where(AuthSession.id != keep)
    await db.execute(q.values(revoked_at=utcnow(), revoke_reason=reason))


@router.post("/refresh")
async def refresh(request: Request, response: Response, db: DB):
    """Silent refresh: the httpOnly cookie's refresh token for a new access token. The refresh token is rotated
    every time; presenting one that was already rotated away (outside a short grace) revokes the session."""
    _check_origin(request)
    token = request.cookies.get(settings.refresh_cookie_name)
    if not token:
        raise HTTPException(401, "نشست پیدا نشد؛ دوباره وارد شوید.")
    digest, now = refresh_hash(token), utcnow()
    session = await db.scalar(select(AuthSession).where(AuthSession.token_hash == digest))
    grace = False
    if not session:
        session = await db.scalar(select(AuthSession).where(AuthSession.prev_token_hash == digest))
        if not session:
            _clear_cookie(response)
            raise HTTPException(401, "نشست نامعتبر است؛ دوباره وارد شوید.")
        if session.revoked_at is None and aware(session.rotated_at) > now - timedelta(seconds=ROTATION_GRACE_SECONDS):
            grace = True
        else:
            if session.revoked_at is None:
                log.warning("refresh token reuse on session %s (user %s): revoked", session.id, session.user_id)
                session.revoked_at, session.revoke_reason = now, "reuse"
                await db.commit()
            _clear_cookie(response)
            raise HTTPException(401, "نشست شما بسته شد؛ دوباره وارد شوید.")
    if session.revoked_at is not None or aware(session.expires_at) < now or aware(session.absolute_expires_at) < now:
        _clear_cookie(response)
        raise HTTPException(401, "نشست شما منقضی شده؛ دوباره وارد شوید.")
    user = await db.get(User, session.user_id)
    if not user or not user.is_active:
        _clear_cookie(response)
        raise HTTPException(401, "حساب کاربری غیرفعال است.")
    if grace:
        # The other tab already rotated and the browser holds its new cookie; just hand out an access token.
        return _access(user, session)
    new_token, new_digest = new_refresh_token()
    expires = min(now + timedelta(days=settings.refresh_idle_days), aware(session.absolute_expires_at))
    done = await db.execute(update(AuthSession).where(AuthSession.id == session.id, AuthSession.token_hash == digest)
                            .values(token_hash=new_digest, prev_token_hash=digest, rotated_at=now, last_used_at=now,
                                    expires_at=expires))
    await db.commit()
    if done.rowcount == 0:  # another request rotated it between our read and write
        return _access(user, session)
    _set_cookie(response, new_token)
    return _access(user, session)


@router.post("/logout")
async def logout(request: Request, response: Response, db: DB):
    _check_origin(request)
    token = request.cookies.get(settings.refresh_cookie_name)
    if token:
        digest = refresh_hash(token)
        session = await db.scalar(select(AuthSession).where(
            (AuthSession.token_hash == digest) | (AuthSession.prev_token_hash == digest)))
        if session and session.revoked_at is None:
            session.revoked_at, session.revoke_reason = utcnow(), "logout"
            await db.commit()
    _clear_cookie(response)
    return {"ok": True}


@router.post("/session/upgrade")
async def upgrade(request: Request, response: Response, db: DB):
    """A browser still holding the old 24-hour token (before sessions) trades it for a session once, so nobody
    is signed out by this change."""
    _check_origin(request)
    auth = request.headers.get("authorization") or ""
    try:
        payload = decode_legacy_token(auth.removeprefix("Bearer ").strip())
    except Exception:
        raise HTTPException(401, "نشست منقضی شده؛ دوباره وارد شوید.")
    user = await db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(401, "حساب کاربری غیرفعال است.")
    return await start_session(db, request, response, user, "upgrade")


# ---------- sign in ----------


class PhoneIn(BaseModel):
    phone: str


class VerifyIn(BaseModel):
    phone: str
    code: str


class PasswordLoginIn(BaseModel):
    phone: str
    password: str = Field(max_length=200)


@router.post("/lookup")
async def lookup(body: PhoneIn, request: Request, db: DB):
    """Which sign-in to show first: password for accounts that have one, otherwise a code by SMS."""
    ratelimit.hit(f"lookup:{ratelimit.client_ip(request)}", 30, 600)
    phone = normalize_phone(body.phone)
    user = await db.scalar(select(User).where(User.phone == phone))
    has_password = bool(user and user.is_active and user.password_hash)
    return {"exists": user is not None, "has_password": has_password, "method": "password" if has_password else "otp"}


@router.post("/otp/send")
async def send_otp(body: PhoneIn, request: Request, db: DB):
    phone = normalize_phone(body.phone)
    ratelimit.hit(f"otp:{ratelimit.client_ip(request)}", 10, 600)
    recent = await db.scalar(select(func.count()).select_from(OtpCode).where(
        OtpCode.phone == phone, OtpCode.created_at > utcnow() - timedelta(minutes=10)))
    if recent >= settings.otp_max_sends_per_10min:
        raise HTTPException(429, "تعداد درخواست کد زیاد است؛ چند دقیقه بعد دوباره تلاش کنید.")
    throttle = await db.get(OtpThrottle, phone)
    if throttle and throttle.locked_until and aware(throttle.locked_until) > utcnow():
        left = (aware(throttle.locked_until) - utcnow()).total_seconds()
        raise _wait(_lock_message(left), left)
    cfg = await sms_config(db)
    if not cfg["ready"] and not settings.otp_dev_mode:
        raise HTTPException(503, "ارسال پیامک هنوز راه‌اندازی نشده است.")
    code = f"{random.SystemRandom().randint(0, 999999):06d}"
    otp = OtpCode(phone=phone, code_hash=hash_code(phone, code),
                  expires_at=utcnow() + timedelta(seconds=settings.otp_ttl_seconds))
    db.add(otp)
    await db.commit()
    if cfg["ready"]:
        try:
            await send_code(cfg, phone, code)
        except SmsError as e:
            log.error("SMS to %s failed: %s", phone[:4] + "*****" + phone[-2:], e)
            otp.used = True  # never delivered: don't let it count as a live code
            await db.commit()
            raise HTTPException(502, "ارسال پیامک انجام نشد؛ کمی بعد دوباره تلاش کنید.")
        return {"sent": True, "ttl": settings.otp_ttl_seconds}
    log.warning("OTP for %s: %s", phone, code)
    return {"sent": True, "ttl": settings.otp_ttl_seconds, "dev_code": code}


@router.post("/otp/verify")
async def verify_otp(body: VerifyIn, request: Request, response: Response, db: DB):
    phone = normalize_phone(body.phone)
    now = utcnow()
    throttle = await db.get(OtpThrottle, phone)
    if throttle and throttle.locked_until and aware(throttle.locked_until) > now:
        left = (aware(throttle.locked_until) - now).total_seconds()
        raise _wait(_lock_message(left), left)
    if throttle and throttle.next_try_at and aware(throttle.next_try_at) > now:
        left = (aware(throttle.next_try_at) - now).total_seconds()
        raise _wait(f"کد را اشتباه وارد کردی؛ {int(left) + 1} ثانیهٔ دیگر دوباره امتحان کن.", left)
    otp = (await db.execute(select(OtpCode).where(OtpCode.phone == phone, ~OtpCode.used)
                            .order_by(OtpCode.id.desc()).limit(1))).scalar_one_or_none()
    if not otp or aware(otp.expires_at) < utcnow():
        raise HTTPException(400, "کد منقضی شده؛ دوباره درخواست کد بدهید.")
    if otp.attempts >= 5:
        raise HTTPException(429, "تعداد تلاش زیاد بود؛ کد جدید بگیرید.")
    otp.attempts += 1
    if otp.code_hash != hash_code(phone, to_en_digits(body.code).strip()):
        if not throttle:
            throttle = OtpThrottle(phone=phone, fails=0)
            db.add(throttle)
        throttle.fails = (throttle.fails or 0) + 1
        if throttle.fails >= OTP_MAX_FAILS:
            throttle.fails, throttle.next_try_at = 0, None
            throttle.locked_until = now + timedelta(seconds=OTP_LOCK_SECONDS)
            await db.commit()
            raise _wait(_lock_message(OTP_LOCK_SECONDS), OTP_LOCK_SECONDS)
        throttle.next_try_at = now + timedelta(seconds=OTP_RETRY_WAIT_SECONDS)
        await db.commit()
        raise _wait("کد وارد شده اشتباه است.", OTP_RETRY_WAIT_SECONDS, 400)
    if throttle:
        throttle.fails, throttle.next_try_at, throttle.locked_until = 0, None, None
    otp.used = True
    user = (await db.execute(select(User).where(User.phone == phone))).scalar_one_or_none()
    is_new = user is None
    if is_new:
        user = User(phone=phone, role="admin" if phone in settings.admin_phones else "seller", is_active=True)
        db.add(user)
    if not user.is_active:
        await db.commit()
        raise HTTPException(403, "حساب کاربری شما غیرفعال شده است.")
    # A code proves the phone: it also clears a password lock.
    user.failed_logins, user.locked_until = 0, None
    await db.flush()
    return {**await start_session(db, request, response, user, "otp"), "is_new": is_new}


@router.post("/password/login")
async def password_login(body: PasswordLoginIn, request: Request, response: Response, db: DB):
    ratelimit.hit(f"pw:{ratelimit.client_ip(request)}", 20, 600)
    phone = normalize_phone(body.phone)
    user = await db.scalar(select(User).where(User.phone == phone))
    now = utcnow()
    if user and user.locked_until and aware(user.locked_until) > now:
        minutes = max(1, int((aware(user.locked_until) - now).total_seconds() // 60) + 1)
        raise HTTPException(429, f"به‌خاطر تلاش‌های ناموفق، ورود با رمز تا {minutes} دقیقه بسته است؛ با کد پیامکی وارد شوید.")
    if not verify_password(body.password, user.password_hash if user else None):
        if user and user.password_hash:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= MAX_PASSWORD_FAILS:
                user.failed_logins, user.locked_until = 0, now + timedelta(minutes=LOCK_MINUTES)
            await db.commit()
        raise HTTPException(400, "شماره یا رمز اشتباه است.")
    if not user.is_active:
        raise HTTPException(403, "حساب کاربری شما غیرفعال شده است.")
    user.failed_logins, user.locked_until = 0, None
    return {**await start_session(db, request, response, user, "password"), "is_new": False}


class SetPasswordIn(BaseModel):
    password: str = Field(max_length=200)
    current_password: str | None = Field(None, max_length=200)


@router.post("/password")
async def set_password(body: SetPasswordIn, user: CurrentUser, session: CurrentSession, db: DB):
    """Sets or changes the password. Changing one needs the current password, unless this device signed in with
    an SMS code in the last 15 minutes (the «forgot password» path). Other devices are signed out."""
    if problem := password_problem(body.password):
        raise HTTPException(422, problem)
    if user.password_hash:
        fresh_otp = session.method == "otp" and aware(session.created_at) > utcnow() - timedelta(minutes=15)
        if not fresh_otp and not verify_password(body.current_password or "", user.password_hash):
            raise HTTPException(400, "رمز فعلی اشتباه است.")
    user.password_hash, user.password_set_at = hash_password(body.password), utcnow()
    user.failed_logins, user.locked_until = 0, None
    await revoke_sessions(db, user.id, "password_changed", keep=session.id)
    await db.commit()
    return {"ok": True, "has_password": True}


class ProfileIn(BaseModel):
    name: str | None = None
    onboarded: bool | None = None


@router.get("/me")
async def me(user: CurrentUser, db: DB):
    # features: what the admin has opened to this user (admin panel › قابلیت‌ها), so the panel shows only those.
    return {**UserOut.model_validate(user).model_dump(), "has_password": bool(user.password_hash),
            "features": await user_features(db, user)}


@router.patch("/me", response_model=UserOut)
async def update_me(body: ProfileIn, user: CurrentUser, db: DB):
    if body.name is not None:
        user.name = body.name
    if body.onboarded is not None:
        user.onboarded = body.onboarded
    await db.commit()
    await db.refresh(user)
    return user
