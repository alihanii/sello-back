from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import DB, CurrentUser, get_store_for
from app.models import AiProductRequest
from app.schemas import OperationOut
from app.services.ai import route_problem
from app.services.ops import enqueue
from app.services.product_images import photo_kind

router = APIRouter(tags=["ai"])


class AiProductIn(BaseModel):
    images: list[str] = []
    note: str = ""
    price: int = 0
    stock_qty: int = 0
    # Clean the first photo, or generate the product photo from the text (and the design) when it isn't one.
    clean_image: bool = False
    photo_kind: str = "product"  # product: clean the photo · design: artwork to put on the product


@router.post("/stores/{store_id}/ai-products", response_model=OperationOut, status_code=202)
async def create(store_id: int, body: AiProductIn, user: CurrentUser, db: DB):
    """One product from the seller's photos and description."""
    await get_store_for(db, user, store_id)
    if not (body.images or body.note.strip()):
        raise HTTPException(422, "حداقل یک عکس یا یک توضیح بدهید.")
    if body.clean_image and (problem := await route_problem("image_edit")):
        raise HTTPException(422, problem)
    req = AiProductRequest(store_id=store_id, images=body.images, note=body.note, price=body.price, stock_qty=body.stock_qty)
    db.add(req)
    await db.flush()
    op = await enqueue(db, store_id=store_id, type="ai_product", ref_type="ai_product_request", ref_id=req.id, actor="ai",
                       title_fa="ساخت محصول با AI",
                       payload={"clean_image": body.clean_image,
                                "photo_kind": "design" if body.photo_kind == "design" else "product"})
    req.operation_id = op.id
    await db.commit()
    return op


class PhotoKindIn(BaseModel):
    url: str


@router.post("/stores/{store_id}/photo-kind")
async def guess_photo_kind(store_id: int, body: PhotoKindIn, user: CurrentUser, db: DB):
    """First guess for the AI form's photo choice; the seller can change it."""
    await get_store_for(db, user, store_id)
    return {"kind": await photo_kind(store_id, body.url)}
