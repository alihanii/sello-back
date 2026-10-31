from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserOut(ORM):
    id: int
    phone: str
    name: str
    role: str
    onboarded: bool
    created_at: datetime
    last_login_at: datetime | None = None


class StoreIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: str = ""
    logo_url: str = ""
    settings: dict[str, Any] | None = None


class StoreOut(ORM):
    id: int
    name: str
    category: str
    logo_url: str
    settings: dict[str, Any]
    archived: bool
    created_at: datetime


class ChannelOut(ORM):
    id: int
    store_id: int
    platform: str
    display_name: str
    external_account_id: str
    config: dict[str, Any]
    status: str
    last_health_at: datetime | None
    last_error: str
    created_at: datetime


class ImageOut(ORM):
    id: int
    url: str
    position: int


class ListingOut(ORM):
    id: int
    channel_id: int
    status: str
    remote_id: str
    remote_url: str
    published_by: str
    published_at: datetime | None
    last_error_fa: str
    updated_at: datetime


class ProductIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = ""
    price: int = 0
    compare_at_price: int = 0
    stock_qty: int = 0
    sku: str = ""
    brand: str = ""
    category_hint: str = ""
    attributes: dict[str, Any] = {}
    tags: list[str] = []
    status: str = "ready"
    images: list[str] = []


class ProductPatch(BaseModel):
    title: str | None = None
    description: str | None = None
    price: int | None = None
    compare_at_price: int | None = None
    stock_qty: int | None = None
    sku: str | None = None
    brand: str | None = None
    category_hint: str | None = None
    attributes: dict[str, Any] | None = None
    tags: list[str] | None = None
    status: str | None = None
    images: list[str] | None = None


class ProductOut(ORM):
    id: int
    store_id: int
    sku: str
    title: str
    description: str
    price: int
    compare_at_price: int
    stock_qty: int
    brand: str
    category_hint: str
    attributes: dict[str, Any]
    source: str
    source_ref: dict[str, Any]
    status: str
    created_by: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    images: list[ImageOut] = []
    listings: list[ListingOut] = []


class StepOut(ORM):
    id: int
    order: int
    name: str
    label_fa: str
    status: str
    started_at: datetime | None
    duration_ms: int
    request: Any
    response: Any
    message_fa: str


class OperationOut(ORM):
    id: int
    store_id: int
    type: str
    title_fa: str
    channel_id: int | None
    product_id: int | None
    ref_type: str
    ref_id: int | None
    trigger: str
    actor: str
    status: str
    current_step: str
    progress: int
    error_code: str
    error_message_fa: str
    suggested_action: dict[str, Any]
    result: dict[str, Any]
    attempt: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    seen: bool


class OperationDetail(OperationOut):
    steps: list[StepOut] = []


class PlanIn(BaseModel):
    name: str
    tier: str = ""
    duration_months: int
    price_toman: int
    compare_at_toman: int = 0
    description: str = ""
    features: list[str] = []
    limits: dict[str, Any] = {}
    is_featured: bool = False
    is_active: bool = True
    sort_order: int = 0


class PlanOut(PlanIn, ORM):
    id: int


class PlanPrice(BaseModel):
    duration_months: int
    price_toman: int
    compare_at_toman: int = 0
    is_active: bool = True
    plan_id: int | None = None


class PlanConfig(BaseModel):
    name: str
    description: str = ""
    features: list[str] = []
    limits: dict[str, Any] = {}
    prices: list[PlanPrice]
