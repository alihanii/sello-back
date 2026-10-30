from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.deps import DB, CurrentUser, get_store_for
from app.models import Approval, Automation, Channel, Product
from app.services.automation import queue_publish

router = APIRouter(tags=["automations"])
TRIGGERS = {"product_created", "product_updated", "stock_changed", "schedule"}


class AutomationIn(BaseModel):
    name: str
    trigger: str
    conditions: dict = {}
    actions: dict = {}
    is_active: bool = True


def _out(a: Automation) -> dict:
    return {c: getattr(a, c) for c in ("id", "store_id", "name", "trigger", "conditions", "actions", "is_active",
                                         "last_run_at", "run_count", "created_at")}


@router.get("/stores/{store_id}/automations")
async def list_rules(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    return [_out(a) for a in (await db.execute(select(Automation).where(Automation.store_id == store_id)
                                                .order_by(Automation.id))).scalars()]


@router.post("/stores/{store_id}/automations")
async def create_rule(store_id: int, body: AutomationIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    if body.trigger not in TRIGGERS:
        raise HTTPException(422, "رویداد نامعتبر است.")
    if not body.actions.get("channel_ids"):
        raise HTTPException(422, "حداقل یک کانال مقصد انتخاب کنید.")
    a = Automation(store_id=store_id, **body.model_dump())
    db.add(a)
    await db.commit()
    return _out(a)


@router.patch("/automations/{rule_id}")
async def update_rule(rule_id: int, body: AutomationIn, user: CurrentUser, db: DB):
    a = await db.get(Automation, rule_id)
    if not a:
        raise HTTPException(404, "اتوماسیون پیدا نشد.")
    await get_store_for(db, user, a.store_id)
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    await db.commit()
    return _out(a)


@router.delete("/automations/{rule_id}")
async def delete_rule(rule_id: int, user: CurrentUser, db: DB):
    a = await db.get(Automation, rule_id)
    if not a:
        raise HTTPException(404, "اتوماسیون پیدا نشد.")
    await get_store_for(db, user, a.store_id)
    await db.delete(a)
    await db.commit()
    return {"ok": True}


@router.get("/stores/{store_id}/approvals")
async def approvals(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(Approval, Product).join(Product, Product.id == Approval.product_id)
                             .where(Approval.store_id == store_id, Approval.status == "pending")
                             .order_by(Approval.id.desc()))).all()
    return [{"id": a.id, "product_id": p.id, "product_title": p.title, "channel_ids": a.channel_ids,
             "automation_id": a.automation_id, "created_at": a.created_at} for a, p in rows]


class Decide(BaseModel):
    approve: bool


@router.post("/approvals/{approval_id}")
async def decide(approval_id: int, body: Decide, user: CurrentUser, db: DB):
    a = await db.get(Approval, approval_id)
    if not a or a.status != "pending":
        raise HTTPException(404, "مورد پیدا نشد.")
    await get_store_for(db, user, a.store_id)
    a.status = "approved" if body.approve else "rejected"
    if body.approve:
        product = await db.get(Product, a.product_id)
        for cid in a.channel_ids:
            ch = await db.get(Channel, cid)
            if ch and ch.status == "connected":
                await queue_publish(db, product, ch, trigger="automation", actor="ai")
    await db.commit()
    return {"ok": True}
