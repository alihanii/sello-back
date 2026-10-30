from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.deps import DB, CurrentUser, get_store_for
from app.models import Approval, Automation, Channel, Product
from app.core.db import utcnow
from app.services.automation import enqueue_ai_posts, queue_publish

router = APIRouter(tags=["automations"])
TRIGGERS = {"product_created", "product_updated", "stock_changed", "schedule", "ai_content"}


class AutomationIn(BaseModel):
    name: str
    trigger: str
    conditions: dict = {}
    actions: dict = {}
    is_active: bool = True


def _validate(body: AutomationIn) -> None:
    if body.trigger not in TRIGGERS:
        raise HTTPException(422, "رویداد نامعتبر است.")
    if not body.actions.get("channel_ids"):
        raise HTTPException(422, "حداقل یک کانال مقصد انتخاب کنید.")
    if body.trigger == "ai_content":
        if not body.actions.get("weekdays"):
            raise HTTPException(422, "حداقل یک روز هفته برای ساخت پست انتخاب کنید.")
        if body.actions.get("source") == "references" and not body.actions.get("queue"):
            raise HTTPException(422, "برای ساخت از رفرنس، حداقل یک عکس یا لینک به صف اضافه کنید.")


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
    _validate(body)
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
    _validate(body)
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


@router.post("/automations/{rule_id}/run")
async def run_now(rule_id: int, user: CurrentUser, db: DB):
    """Try a content automation immediately instead of waiting for its next scheduled day."""
    a = await db.get(Automation, rule_id)
    if not a:
        raise HTTPException(404, "اتوماسیون پیدا نشد.")
    await get_store_for(db, user, a.store_id)
    if a.trigger != "ai_content":
        raise HTTPException(422, "فقط اتوماسیون‌های محتوای AI را می‌شود دستی اجرا کرد.")
    ids = await enqueue_ai_posts(db, a, trigger="manual", actor="user")
    await db.commit()
    return {"operation_ids": ids}


@router.get("/stores/{store_id}/approvals")
async def approvals(store_id: int, user: CurrentUser, db: DB, status: str = "pending"):
    await get_store_for(db, user, store_id)
    rows = (await db.execute(select(Approval, Product, Automation)
                             .join(Product, Product.id == Approval.product_id)
                             .outerjoin(Automation, Automation.id == Approval.automation_id)
                             .where(Approval.store_id == store_id, Approval.status == status)
                             .options(selectinload(Product.images))
                             .order_by(Approval.id.desc()).limit(100))).all()
    return [{"id": ap.id, "kind": ap.kind, "status": ap.status, "product_id": p.id, "product_title": p.title,
             "price": p.price, "caption": p.post_caption, "description": p.description,
             "images": [i.url for i in p.images[:6]], "channel_ids": ap.channel_ids, "automation_id": ap.automation_id,
             "automation_name": r.name if r else None, "operation_id": ap.operation_id, "created_at": ap.created_at}
            for ap, p, r in rows]


class Decide(BaseModel):
    approve: bool
    caption: str | None = None  # the owner may edit the AI text before approving
    channel_ids: list[int] | None = None


@router.post("/approvals/{approval_id}")
async def decide(approval_id: int, body: Decide, user: CurrentUser, db: DB):
    a = await db.get(Approval, approval_id)
    if not a or a.status != "pending":
        raise HTTPException(404, "این مورد قبلاً بررسی شده یا پیدا نشد.")
    await get_store_for(db, user, a.store_id)
    a.status, a.decided_at = ("approved" if body.approve else "rejected"), utcnow()
    queued = []
    if body.approve:
        product = await db.get(Product, a.product_id)
        if body.caption is not None:
            product.post_caption = body.caption.strip()
        if body.channel_ids is not None:
            a.channel_ids = body.channel_ids
        for cid in a.channel_ids:
            ch = await db.get(Channel, cid)
            if ch and ch.store_id == a.store_id and ch.status == "connected":
                op_id = await queue_publish(db, product, ch, trigger="automation", actor="ai")
                if op_id:
                    queued.append(op_id)
    await db.commit()
    return {"ok": True, "operation_ids": queued}
