from fastapi import APIRouter
from sqlalchemy import select

from app.deps import DB, CurrentUser, get_store_for
from app.models import DEFAULT_STORE_SETTINGS, Store
from app.schemas import StoreIn, StoreOut
from app.services.plans import check_limit

router = APIRouter(prefix="/stores", tags=["stores"])


@router.get("", response_model=list[StoreOut])
async def list_stores(user: CurrentUser, db: DB):
    q = select(Store).where(Store.owner_id == user.id, ~Store.archived).order_by(Store.id)
    return (await db.execute(q)).scalars().all()


@router.post("", response_model=StoreOut)
async def create_store(body: StoreIn, user: CurrentUser, db: DB):
    await check_limit(db, user, "stores")
    store = Store(owner_id=user.id, name=body.name, category=body.category, logo_url=body.logo_url,
                  settings={**DEFAULT_STORE_SETTINGS, **(body.settings or {})})
    db.add(store)
    await db.commit()
    await db.refresh(store)
    return store


@router.get("/{store_id}", response_model=StoreOut)
async def get_store(store_id: int, user: CurrentUser, db: DB):
    return await get_store_for(db, user, store_id)


@router.patch("/{store_id}", response_model=StoreOut)
async def update_store(store_id: int, body: StoreIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    store.name, store.category, store.logo_url = body.name, body.category, body.logo_url
    if body.settings is not None:
        store.settings = {**DEFAULT_STORE_SETTINGS, **store.settings, **body.settings}
    await db.commit()
    await db.refresh(store)
    return store


@router.delete("/{store_id}")
async def archive_store(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    store.archived = True
    await db.commit()
    return {"ok": True}
