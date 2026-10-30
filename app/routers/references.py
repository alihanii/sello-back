from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.deps import DB, CurrentUser, get_store_for
from app.models import ReferenceSet, StoreStyleProfile
from app.schemas import OperationOut
from app.services.ai import route_problem
from app.services.ops import enqueue

router = APIRouter(tags=["ai"])


class ReferenceIn(BaseModel):
    images: list[str] = []
    product_ids: list[int] = []
    urls: list[str] = []
    note: str = ""
    price: int = 0
    stock_qty: int = 0
    use_style_profile: bool = True
    # One product per image/link instead of one product from all references.
    bulk: bool = False
    # Clean the first photo, or generate one from the text when there is none, in the shop's photo style.
    clean_image: bool = False
    # Style samples for that photo (uploaded photos / existing products); empty = the shop's earlier products.
    style_images: list[str] = []
    style_product_ids: list[int] = []


@router.post("/stores/{store_id}/reference-sets", response_model=list[OperationOut], status_code=202)
async def create(store_id: int, body: ReferenceIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    if not (body.images or body.urls or body.note.strip()):
        raise HTTPException(422, "حداقل یک عکس، لینک یا توضیح بدهید.")
    if body.clean_image and (problem := await route_problem("image_edit")):
        raise HTTPException(422, problem)
    groups = ([{"images": [i], "urls": []} for i in body.images] + [{"images": [], "urls": [u]} for u in body.urls]
              if body.bulk else [{"images": body.images, "urls": body.urls}])
    ops = []
    for g in groups:
        ref = ReferenceSet(store_id=store_id, images=g["images"], urls=g["urls"], product_ids=body.product_ids,
                           note=body.note, price=body.price, stock_qty=body.stock_qty,
                           use_style_profile=body.use_style_profile)
        db.add(ref)
        await db.flush()
        op = await enqueue(db, store_id=store_id, type="reference", ref_type="reference_set", ref_id=ref.id, actor="ai",
                           title_fa="ساخت محصول با AI از رفرنس",
                           payload={"clean_image": body.clean_image, "style_images": body.style_images[:3],
                                    "style_product_ids": body.style_product_ids[:3]})
        ref.operation_id = op.id
        ops.append(op)
    await db.commit()
    return ops


@router.get("/stores/{store_id}/style-profile")
async def get_profile(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    p = (await db.execute(select(StoreStyleProfile).where(StoreStyleProfile.store_id == store_id))).scalar_one_or_none()
    return {"profile": p.profile if p else None, "sample_product_ids": p.sample_product_ids if p else [],
            "updated_at": p.updated_at if p else None}


class ProfileIn(BaseModel):
    profile: dict


@router.put("/stores/{store_id}/style-profile")
async def put_profile(store_id: int, body: ProfileIn, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    p = (await db.execute(select(StoreStyleProfile).where(StoreStyleProfile.store_id == store_id))).scalar_one_or_none()
    if not p:
        p = StoreStyleProfile(store_id=store_id)
        db.add(p)
    p.profile = body.profile
    await db.commit()
    return {"ok": True}


@router.post("/stores/{store_id}/style-profile/rebuild", response_model=OperationOut, status_code=202)
async def rebuild(store_id: int, user: CurrentUser, db: DB):
    await get_store_for(db, user, store_id)
    op = await enqueue(db, store_id=store_id, type="style_profile", actor="ai", title_fa="ساخت پروفایل سبک فروشگاه")
    await db.commit()
    return op
