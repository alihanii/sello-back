from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.deps import DB, CurrentUser
from app.models import FeatureInterest, Plan, PlanRequest
from app.schemas import PlanOut
from app.services.plans import active_subscription, limits_for

router = APIRouter(tags=["billing"])


@router.get("/plans", response_model=list[PlanOut])
async def plans(db: DB):
    q = select(Plan).where(Plan.is_active).order_by(Plan.sort_order, Plan.duration_months, Plan.price_toman)
    return (await db.execute(q)).scalars().all()


@router.get("/me/subscription")
async def my_subscription(user: CurrentUser, db: DB):
    sub = await active_subscription(db, user.id)
    plan = await db.get(Plan, sub.plan_id) if sub else None
    pending = (await db.execute(select(PlanRequest).where(PlanRequest.user_id == user.id, PlanRequest.status == "pending")
                                .order_by(PlanRequest.id.desc()))).scalars().first()
    return {
        "subscription": {"plan": PlanOut.model_validate(plan), "starts_at": sub.starts_at, "ends_at": sub.ends_at}
        if sub else None,
        "limits": await limits_for(db, user),
        "pending_request": {"id": pending.id, "plan_id": pending.plan_id, "created_at": pending.created_at} if pending else None,
    }


class RequestIn(BaseModel):
    plan_id: int
    note: str = ""


@router.post("/plan-requests")
async def request_plan(body: RequestIn, user: CurrentUser, db: DB):
    plan = await db.get(Plan, body.plan_id)
    if not plan or not plan.is_active:
        raise HTTPException(404, "پلن پیدا نشد.")
    db.add(PlanRequest(user_id=user.id, plan_id=plan.id, note=body.note))
    await db.commit()
    return {"ok": True, "message": "درخواست شما ثبت شد؛ پس از بررسی، پلن فعال می‌شود."}


class InterestIn(BaseModel):
    feature: str


@router.post("/feature-interest")
async def interest(body: InterestIn, user: CurrentUser, db: DB):
    db.add(FeatureInterest(user_id=user.id, feature=body.feature))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
    return {"ok": True}


@router.get("/feature-interest/{feature}")
async def my_interest(feature: str, user: CurrentUser, db: DB):
    row = (await db.execute(select(FeatureInterest).where(FeatureInterest.user_id == user.id,
                                                          FeatureInterest.feature == feature))).scalar_one_or_none()
    return {"registered": row is not None}
