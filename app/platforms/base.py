"""Adapter contract every platform implements (spec §4). The runner passes a `ctx` whose
`step()` records each stage with its request/response for the publish log."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.models import Channel, Product, Store
    from app.services.ops import OpContext


class PlatformError(Exception):
    """Raised with a Persian message the seller can act on."""

    def __init__(self, message_fa: str, code: str = "platform_error", suggested_action: dict | None = None,
                 detail: Any = None):
        super().__init__(message_fa)
        self.message_fa = message_fa
        self.code = code
        self.suggested_action = suggested_action or {}
        self.detail = detail


RECONNECT = {"type": "reconnect", "label": "اتصال مجدد"}
EDIT_PRODUCT = {"type": "edit_product", "label": "ویرایش محصول"}


@dataclass
class PublishResult:
    remote_id: str
    remote_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawledItem:
    external_id: str
    caption: str = ""
    title: str = ""
    description: str = ""
    price: int = 0
    quantity: int = 0
    sku: str = ""
    brand: str = ""
    category: str = ""
    image_urls: list[str] = field(default_factory=list)
    url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    platform: str
    label_fa: str

    async def connect(self, ctx: "OpContext", channel: "Channel", data: dict[str, Any]) -> dict[str, Any]: ...
    async def health(self, ctx: "OpContext", channel: "Channel") -> None: ...
    def validate(self, product: "Product") -> list[str]: ...
    async def publish(self, ctx: "OpContext", channel: "Channel", product: "Product", store: "Store",
                      existing_remote_id: str = "") -> PublishResult: ...
    async def fetch_items(self, ctx: "OpContext", channel: "Channel", limit: int = 50) -> list[CrawledItem]: ...
