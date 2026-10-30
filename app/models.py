"""All Sello tables. Grouped by area; see the phase one spec, section 5."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, TimestampMixin

# ---------- accounts ----------


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    role: Mapped[str] = mapped_column(String(20), default="seller")  # seller | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    stores: Mapped[list["Store"]] = relationship(back_populates="owner")


class OtpCode(TimestampMixin, Base):
    __tablename__ = "otp_codes"
    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(20), index=True)
    code_hash: Mapped[str] = mapped_column(String(128))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    used: Mapped[bool] = mapped_column(Boolean, default=False)


# ---------- stores & channels ----------


class Store(TimestampMixin, Base):
    __tablename__ = "stores"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(80), default="")
    logo_url: Mapped[str] = mapped_column(String(500), default="")
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)

    owner: Mapped[User] = relationship(back_populates="stores")
    channels: Mapped[list["Channel"]] = relationship(back_populates="store")


DEFAULT_STORE_SETTINGS: dict[str, Any] = {
    "show_price_in_caption": True,
    "caption_signature": "",
    "ai_tone": "friendly",  # friendly | formal
    "default_publish_mode": "approval",  # auto | approval
    "low_stock_threshold": 3,
    "crawl_to_inbox": False,
}


class Channel(TimestampMixin, Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    platform: Mapped[str] = mapped_column(String(20))  # telegram | bale | digikala | instagram
    display_name: Mapped[str] = mapped_column(String(160), default="")
    external_account_id: Mapped[str] = mapped_column(String(160), default="")
    credentials_encrypted: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="disconnected")  # connected|expired|error|disconnected
    last_health_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str] = mapped_column(Text, default="")

    store: Mapped[Store] = relationship(back_populates="channels")


# ---------- products ----------


class Product(TimestampMixin, Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    sku: Mapped[str] = mapped_column(String(80), default="")
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[int] = mapped_column(BigInteger, default=0)  # Toman
    compare_at_price: Mapped[int] = mapped_column(BigInteger, default=0)
    stock_qty: Mapped[int] = mapped_column(Integer, default=0)
    brand: Mapped[str] = mapped_column(String(120), default="")
    category_hint: Mapped[str] = mapped_column(String(200), default="")
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(10), default="manual")  # manual | crawl | ai
    source_ref: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(12), default="ready")  # draft | ready | archived
    created_by: Mapped[str] = mapped_column(String(10), default="user")  # user | ai | crawler
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_golden_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    # Ready-to-post text written by AI (content automations); channels use it instead of title + description.
    post_caption: Mapped[str] = mapped_column(Text, default="")

    images: Mapped[list["ProductImage"]] = relationship(
        back_populates="product", order_by="ProductImage.position", cascade="all, delete-orphan"
    )
    listings: Mapped[list["ChannelListing"]] = relationship(back_populates="product", cascade="all, delete-orphan")


class ProductImage(TimestampMixin, Base):
    __tablename__ = "product_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    url: Mapped[str] = mapped_column(String(1000))
    position: Mapped[int] = mapped_column(Integer, default=0)
    origin_url: Mapped[str] = mapped_column(String(1000), default="")

    product: Mapped[Product] = relationship(back_populates="images")


class InventoryMovement(TimestampMixin, Base):
    __tablename__ = "inventory_movements"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    delta: Mapped[int] = mapped_column(Integer)
    qty_after: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(20), default="manual")  # manual | crawl_sync | channel_sync
    note: Mapped[str] = mapped_column(String(300), default="")
    actor: Mapped[str] = mapped_column(String(20), default="user")


class ChannelListing(TimestampMixin, Base):
    """Product × channel. The unique key is what stops duplicate publishes."""

    __tablename__ = "channel_listings"
    __table_args__ = (UniqueConstraint("product_id", "channel_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"), index=True)
    # not_published | queued | publishing | published | failed | out_of_sync | removed
    status: Mapped[str] = mapped_column(String(16), default="not_published")
    remote_id: Mapped[str] = mapped_column(String(200), default="")
    remote_url: Mapped[str] = mapped_column(String(1000), default="")
    published_by: Mapped[str] = mapped_column(String(12), default="")  # user | ai | automation
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_fa: Mapped[str] = mapped_column(Text, default="")

    product: Mapped[Product] = relationship(back_populates="listings")
    channel: Mapped[Channel] = relationship()


# ---------- operations (status center), steps, crawl ----------


class Operation(TimestampMixin, Base):
    """One row per thing the system does. Status center and job queue share this table."""

    __tablename__ = "operations"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    # connect | publish | crawl | health_check | automation | reference
    type: Mapped[str] = mapped_column(String(20), index=True)
    title_fa: Mapped[str] = mapped_column(String(300), default="")
    channel_id: Mapped[int | None] = mapped_column(ForeignKey("channels.id"))
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    ref_type: Mapped[str] = mapped_column(String(30), default="")
    ref_id: Mapped[int | None] = mapped_column(Integer)
    trigger: Mapped[str] = mapped_column(String(20), default="user")  # user | automation | schedule | retry
    actor: Mapped[str] = mapped_column(String(12), default="user")  # user | ai
    # queued | running | succeeded | failed | partial
    status: Mapped[str] = mapped_column(String(12), default="queued", index=True)
    current_step: Mapped[str] = mapped_column(String(80), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str] = mapped_column(String(60), default="")
    error_message_fa: Mapped[str] = mapped_column(Text, default="")
    suggested_action: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    seen: Mapped[bool] = mapped_column(Boolean, default=False)

    steps: Mapped[list["OperationStep"]] = relationship(order_by="OperationStep.order", cascade="all, delete-orphan")


class OperationStep(TimestampMixin, Base):
    __tablename__ = "operation_steps"
    id: Mapped[int] = mapped_column(primary_key=True)
    operation_id: Mapped[int] = mapped_column(ForeignKey("operations.id"), index=True)
    order: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(60))
    label_fa: Mapped[str] = mapped_column(String(160), default="")
    status: Mapped[str] = mapped_column(String(12), default="running")  # running | succeeded | failed | skipped
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    request: Mapped[Any] = mapped_column(JSON, default=dict)
    response: Mapped[Any] = mapped_column(JSON, default=dict)
    message_fa: Mapped[str] = mapped_column(Text, default="")


class CrawlSource(TimestampMixin, Base):
    __tablename__ = "crawl_sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    name: Mapped[str] = mapped_column(String(160), default="")
    type: Mapped[str] = mapped_column(String(24))  # channel_posts | marketplace_listing | url
    channel_id: Mapped[int | None] = mapped_column(ForeignKey("channels.id"))
    url: Mapped[str] = mapped_column(String(1000), default="")
    schedule: Mapped[str] = mapped_column(String(20), default="manual")  # manual | hourly | daily | weekly
    extraction_mode: Mapped[str] = mapped_column(String(8), default="auto")  # auto | regex | llm
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CrawlItem(TimestampMixin, Base):
    __tablename__ = "crawl_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    operation_id: Mapped[int] = mapped_column(ForeignKey("operations.id"), index=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("crawl_sources.id"))
    external_id: Mapped[str] = mapped_column(String(200), default="")
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    extracted: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    result: Mapped[str] = mapped_column(String(12), default="")  # created | updated | duplicate | skipped | inbox | failed
    error_fa: Mapped[str] = mapped_column(Text, default="")


class TelegramUpdate(TimestampMixin, Base):
    """Channel posts the bot received; the crawler reads from here (bots can't read history)."""

    __tablename__ = "bot_channel_posts"
    __table_args__ = (UniqueConstraint("channel_id", "message_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"), index=True)
    message_id: Mapped[str] = mapped_column(String(40))
    caption: Mapped[str] = mapped_column(Text, default="")
    photo_file_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------- AI references & style ----------


class ReferenceSet(TimestampMixin, Base):
    __tablename__ = "reference_sets"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    images: Mapped[list[str]] = mapped_column(JSON, default=list)
    product_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    urls: Mapped[list[str]] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[int] = mapped_column(BigInteger, default=0)
    stock_qty: Mapped[int] = mapped_column(Integer, default=0)
    use_style_profile: Mapped[bool] = mapped_column(Boolean, default=True)
    url_fetches: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    result_product_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    operation_id: Mapped[int | None] = mapped_column(ForeignKey("operations.id"))


class StoreStyleProfile(TimestampMixin, Base):
    __tablename__ = "store_style_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), unique=True)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sample_product_ids: Mapped[list[int]] = mapped_column(JSON, default=list)


# ---------- automations ----------


class Automation(TimestampMixin, Base):
    __tablename__ = "automations"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    # product_created | product_updated | stock_changed | schedule | ai_content
    trigger: Mapped[str] = mapped_column(String(20))
    conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # {"channel_ids": [...], "mode": "auto"|"approval", "schedule_hour": 10}
    actions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    run_count: Mapped[int] = mapped_column(Integer, default=0)


class Approval(TimestampMixin, Base):
    __tablename__ = "approvals"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    automation_id: Mapped[int | None] = mapped_column(ForeignKey("automations.id"))
    channel_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending | approved | rejected
    kind: Mapped[str] = mapped_column(String(12), default="publish")  # publish | ai_post
    operation_id: Mapped[int | None] = mapped_column(ForeignKey("operations.id"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Notification(TimestampMixin, Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    store_id: Mapped[int | None] = mapped_column(ForeignKey("stores.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # approval | publish_failed | channel_error | info
    title_fa: Mapped[str] = mapped_column(String(200))
    body_fa: Mapped[str] = mapped_column(Text, default="")
    link: Mapped[str] = mapped_column(String(300), default="")
    image_url: Mapped[str] = mapped_column(String(1000), default="")
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)


# ---------- calendar ----------


class ActivityEvent(TimestampMixin, Base):
    __tablename__ = "activity_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    # crawl_run | crawl_item | product_created | publish_ok | publish_failed | crawl_failed
    type: Mapped[str] = mapped_column(String(24), index=True)
    count: Mapped[int] = mapped_column(Integer, default=1)
    operation_id: Mapped[int | None] = mapped_column(ForeignKey("operations.id"))
    channel_id: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ---------- plans & billing ----------


class Plan(TimestampMixin, Base):
    __tablename__ = "plans"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    tier: Mapped[str] = mapped_column(String(40), default="")  # groups the 1/3/12 month variants of one plan
    duration_months: Mapped[int] = mapped_column(Integer)  # 1 | 3 | 12
    price_toman: Mapped[int] = mapped_column(BigInteger)
    compare_at_toman: Mapped[int] = mapped_column(BigInteger, default=0)
    description: Mapped[str] = mapped_column(String(300), default="")
    features: Mapped[list[str]] = mapped_column(JSON, default=list)
    limits: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class Subscription(TimestampMixin, Base):
    __tablename__ = "subscriptions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plans.id"))
    status: Mapped[str] = mapped_column(String(12), default="active")  # active | expired | cancelled
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    activated_by: Mapped[str] = mapped_column(String(40), default="admin")
    note: Mapped[str] = mapped_column(String(300), default="")

    plan: Mapped[Plan] = relationship()


class PlanRequest(TimestampMixin, Base):
    __tablename__ = "plan_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plans.id"))
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending | approved | rejected
    note: Mapped[str] = mapped_column(String(300), default="")

    user: Mapped[User] = relationship()
    plan: Mapped[Plan] = relationship()


class FeatureInterest(TimestampMixin, Base):
    __tablename__ = "feature_interests"
    __table_args__ = (UniqueConstraint("user_id", "feature"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    feature: Mapped[str] = mapped_column(String(40))


class AdminAuditLog(TimestampMixin, Base):
    __tablename__ = "admin_audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    admin_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(60))
    target_type: Mapped[str] = mapped_column(String(30))
    target_id: Mapped[int | None] = mapped_column(Integer)
    diff: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ---------- AI keys (managed by admin) ----------


class AiProvider(TimestampMixin, Base):
    """An API key the admin added. `kind` decides the wire format."""

    __tablename__ = "ai_providers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    # gemini (Google or MetisAI gateway) | openai (OpenAI or any compatible gateway) | anthropic
    kind: Mapped[str] = mapped_column(String(20))
    base_url: Mapped[str] = mapped_column(String(300), default="")
    api_key_encrypted: Mapped[str] = mapped_column(Text)
    key_hint: Mapped[str] = mapped_column(String(20), default="")  # last 4 chars, for display
    default_model: Mapped[str] = mapped_column(String(120), default="")
    # Per 1M tokens, in Toman; used only for cost estimates on the monitoring page.
    input_price_per_m: Mapped[int] = mapped_column(BigInteger, default=0)
    output_price_per_m: Mapped[int] = mapped_column(BigInteger, default=0)
    monthly_budget_toman: Mapped[int] = mapped_column(BigInteger, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_test_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_test_ok: Mapped[bool | None] = mapped_column(Boolean)
    last_test_message: Mapped[str] = mapped_column(Text, default="")


AI_TASKS: dict[str, str] = {
    "caption_extract": "استخراج محصول از پست/کپشن",
    "product_generate": "ساخت محصول (از رفرنس، عکس، لینک)",
    "vision": "تحلیل عکس محصول",
    "style_profile": "ساخت پروفایل سبک فروشگاه",
    "field_fill": "پر کردن فیلدهای اجباری پلتفرم‌ها",
    "caption_rewrite": "بازنویسی کپشن برای هر کانال",
    "image_edit": "تمیز کردن عکس محصول (پس‌زمینهٔ سفید) — فقط مدل تصویری",
}
# Tasks that need a model which returns an image; every other task needs a text (JSON) model.
IMAGE_TASKS = {"image_edit"}


class AiTaskRoute(TimestampMixin, Base):
    __tablename__ = "ai_task_routes"
    id: Mapped[int] = mapped_column(primary_key=True)
    task: Mapped[str] = mapped_column(String(40), unique=True)
    provider_id: Mapped[int | None] = mapped_column(ForeignKey("ai_providers.id"))
    model: Mapped[str] = mapped_column(String(120), default="")
    fallback_provider_id: Mapped[int | None] = mapped_column(ForeignKey("ai_providers.id"))
    fallback_model: Mapped[str] = mapped_column(String(120), default="")
    temperature: Mapped[int] = mapped_column(Integer, default=30)  # ×100


class AiUsageLog(Base):
    __tablename__ = "ai_usage_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    provider_id: Mapped[int | None] = mapped_column(ForeignKey("ai_providers.id"), index=True)
    task: Mapped[str] = mapped_column(String(40), index=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    store_id: Mapped[int | None] = mapped_column(Integer, index=True)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    is_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_toman: Mapped[int] = mapped_column(BigInteger, default=0)
    error_code: Mapped[str] = mapped_column(String(60), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")


class AppSetting(TimestampMixin, Base):
    """Admin-managed integration settings (e.g. the Meta app for Instagram). Secrets are Fernet-encrypted."""

    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    secret_encrypted: Mapped[str | None] = mapped_column(Text)


class Payment(TimestampMixin, Base):
    """One online purchase of a plan. The gateway round-trip itself is recorded by az-iranian-bank-gateways
    (its own `azbankgateways_bank` table); `tracking_code` links the two."""

    __tablename__ = "payments"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plans.id"))
    amount_toman: Mapped[int] = mapped_column(BigInteger)
    bank_type: Mapped[str] = mapped_column(String(30))
    tracking_code: Mapped[str] = mapped_column(String(64), index=True, default="")
    reference_number: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(12), default="pending")  # pending | paid | failed
    message: Mapped[str] = mapped_column(Text, default="")
    subscription_id: Mapped[int | None] = mapped_column(ForeignKey("subscriptions.id"))
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship()
    plan: Mapped[Plan] = relationship()
