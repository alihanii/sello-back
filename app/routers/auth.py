import logging
import random
import re
from datetime import timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.config import settings
from app.core.db import utcnow
from app.core.security import create_access_token, hash_code
from app.deps import DB, CurrentUser
from app.models import OtpCode, User
from app.schemas import UserOut
from app.services.text import to_en_digits

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("sello.auth")


def normalize_phone(phone: str) -> str:
    d = re.sub(r"\D", "", to_en_digits(phone))
    if d.startswith("98"):
        d = "0" + d[2:]
    if not d.startswith("0"):
        d = "0" + d
    if not re.fullmatch(r"09\d{9}", d):
        raise HTTPException(422, "شمارهٔ موبایل معتبر نیست.")
    return d


class SendIn(BaseModel):
    phone: str


class VerifyIn(BaseModel):
    phone: str
    code: str


@router.post("/otp/send")
async def send_otp(body: SendIn, db: DB):
    phone = normalize_phone(body.phone)
    recent = await db.scalar(select(func.count()).select_from(OtpCode).where(
        OtpCode.phone == phone, OtpCode.created_at > utcnow() - timedelta(minutes=10)))
    if recent >= settings.otp_max_sends_per_10min:
        raise HTTPException(429, "تعداد درخواست کد زیاد است؛ چند دقیقه بعد دوباره تلاش کنید.")
    code = f"{random.SystemRandom().randint(0, 999999):06d}"
    db.add(OtpCode(phone=phone, code_hash=hash_code(phone, code),
                   expires_at=utcnow() + timedelta(seconds=settings.otp_ttl_seconds)))
    await db.commit()
    if settings.otp_dev_mode:
        log.warning("OTP for %s: %s", phone, code)
        return {"sent": True, "ttl": settings.otp_ttl_seconds, "dev_code": code}
    # TODO(sms): send via settings.sms_provider (Kavenegar / SMS.ir)
    return {"sent": True, "ttl": settings.otp_ttl_seconds}


@router.post("/otp/verify")
async def verify_otp(body: VerifyIn, db: DB):
    phone = normalize_phone(body.phone)
    otp = (await db.execute(select(OtpCode).where(OtpCode.phone == phone, ~OtpCode.used)
                            .order_by(OtpCode.id.desc()).limit(1))).scalar_one_or_none()
    if not otp or otp.expires_at.replace(tzinfo=utcnow().tzinfo) < utcnow():
        raise HTTPException(400, "کد منقضی شده؛ دوباره درخواست کد بدهید.")
    if otp.attempts >= 5:
        raise HTTPException(429, "تعداد تلاش زیاد بود؛ کد جدید بگیرید.")
    otp.attempts += 1
    if otp.code_hash != hash_code(phone, to_en_digits(body.code).strip()):
        await db.commit()
        raise HTTPException(400, "کد وارد شده اشتباه است.")
    otp.used = True
    user = (await db.execute(select(User).where(User.phone == phone))).scalar_one_or_none()
    is_new = user is None
    if is_new:
        user = User(phone=phone, role="admin" if phone in settings.admin_phones else "seller", is_active=True)
        db.add(user)
    if not user.is_active:
        raise HTTPException(403, "حساب کاربری شما غیرفعال شده است.")
    user.last_login_at = utcnow()
    await db.commit()
    await db.refresh(user)
    return {"access_token": create_access_token(user.id, user.role), "user": UserOut.model_validate(user),
            "is_new": is_new}


class ProfileIn(BaseModel):
    name: str | None = None
    onboarded: bool | None = None


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return user


@router.patch("/me", response_model=UserOut)
async def update_me(body: ProfileIn, user: CurrentUser, db: DB):
    if body.name is not None:
        user.name = body.name
    if body.onboarded is not None:
        user.onboarded = body.onboarded
    await db.commit()
    await db.refresh(user)
    return user
