"""Online plan purchase: start a gateway payment, then verify it on the bank callback and activate the plan."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import settings
from app.core.db import utcnow
from app.deps import DB, CurrentUser
from app.models import Payment, Plan, Store
from app.services import bank
from app.services.integrations import payment_config
from app.services.notify import notify
from app.services.plans import create_subscription

log = logging.getLogger("sello.payments")
router = APIRouter(tags=["payments"])


@router.get("/payments/config")
async def config(user: CurrentUser, db: DB):
    cfg = await payment_config(db)
    return {"enabled": cfg["enabled"], "bank": cfg["bank"], "label": bank.BANKS.get(cfg["bank"], {}).get("label", "")}


class StartIn(BaseModel):
    plan_id: int


@router.post("/payments/start")
async def start(body: StartIn, user: CurrentUser, db: DB):
    cfg = await payment_config(db)
    if not cfg["enabled"]:
        raise HTTPException(400, "پرداخت آنلاین هنوز فعال نشده است؛ درخواست خرید ثبت کنید.")
    plan = await db.get(Plan, body.plan_id)
    if not plan or not plan.is_active:
        raise HTTPException(404, "پلن پیدا نشد.")
    pay = Payment(user_id=user.id, plan_id=plan.id, amount_toman=plan.price_toman, bank_type=cfg["bank"])
    db.add(pay)
    await db.commit()
    return_url = f"{settings.frontend_base_url.rstrip('/')}/app/billing"
    try:
        res = await asyncio.to_thread(bank.start, cfg, amount_toman=plan.price_toman, mobile=user.phone, return_url=return_url)
    except Exception as exc:  # noqa: BLE001 - any library or bank failure is shown to the buyer
        log.exception("payment start failed")
        pay.status, pay.message = "failed", f"{exc.__class__.__name__}: {exc}"[:1000]
        await db.commit()
        raise HTTPException(502, "اتصال به درگاه پرداخت ناموفق بود؛ کمی بعد دوباره تلاش کنید.")
    pay.tracking_code, pay.reference_number = res["tracking_code"], res["reference_number"] or ""
    await db.commit()
    return {"payment_id": pay.id, **res["gateway"]}


async def _finish(db, query: dict, form: dict, method: str) -> str:
    cfg = await payment_config(db)
    try:
        res = await asyncio.to_thread(bank.verify, cfg, query=query, form=form, method=method)
    except Exception as exc:  # noqa: BLE001
        res = {"tracking_code": "", "success": False, "message": str(exc)}
    pay = (await db.execute(select(Payment).where(Payment.tracking_code == res["tracking_code"]))).scalars().first() \
        if res["tracking_code"] else None
    if not pay:
        return "failed"
    if pay.status == "paid":  # the bank may call back twice
        return f"success&id={pay.id}"
    if not res["success"]:
        pay.status, pay.message = "failed", (res.get("message") or res.get("status") or "")[:1000]
        await db.commit()
        return f"failed&id={pay.id}"
    plan = await db.get(Plan, pay.plan_id)
    sub = await create_subscription(db, pay.user_id, plan, activated_by=f"payment:{pay.id}",
                                    note=f"پرداخت آنلاین {pay.bank_type} · {pay.tracking_code}")
    await db.flush()
    pay.status, pay.paid_at, pay.subscription_id, pay.message = "paid", utcnow(), sub.id, res.get("message", "")[:1000]
    store = (await db.execute(select(Store).where(Store.owner_id == pay.user_id).order_by(Store.id))).scalars().first()
    if store:
        await notify(db, store.id, "info", "اشتراک فعال شد", f"پلن {plan.name} ({plan.duration_months} ماهه) پرداخت و فعال شد.",
                     link="/app/billing")
    await db.commit()
    return f"success&id={pay.id}"


@router.api_route("/payments/callback/", methods=["GET", "POST"])
async def callback(request: Request, db: DB):
    form = dict(await request.form()) if request.method == "POST" else {}
    outcome = await _finish(db, dict(request.query_params), {k: str(v) for k, v in form.items()}, request.method)
    return RedirectResponse(f"{settings.frontend_base_url.rstrip('/')}/app/billing?payment={outcome}", status_code=303)
