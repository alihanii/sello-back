"""Product categories: a plain per-product field. The seller edits it by hand; AI fills it once, when it creates a product."""

import re

from sqlalchemy import func, select

from app.models import Product
from app.services.text import normalize_fa

MAX_LEN = 120
# How many of the shop's categories are shown to the AI, so it reuses them instead of inventing near-duplicates.
PROMPT_LIMIT = 80


def clean(name: object) -> str:
    s = re.sub(r"\s+", " ", normalize_fa(str(name or ""))).strip(" -،,.؛:")
    return s[:MAX_LEN]


def _key(name: str) -> str:
    return clean(name).replace("‌", "").replace(" ", "").lower()


async def store_categories(db, store_id: int) -> list[dict]:
    """Categories in use in the shop, most used first."""
    n = func.count(Product.id)
    rows = (await db.execute(
        select(Product.category, n).where(Product.store_id == store_id, Product.status != "archived",
                                          Product.category.is_not(None), Product.category != "")
        .group_by(Product.category).order_by(n.desc(), Product.category))).all()
    return [{"name": name, "count": count} for name, count in rows]


async def category_names(db, store_id: int) -> list[str]:
    return [c["name"] for c in await store_categories(db, store_id)]


def match(name: object, existing: list[str]) -> str:
    """The AI's answer, snapped to an existing category when it is the same name written differently
    (Arabic ی/ک, spaces, half-spaces); otherwise the cleaned new name."""
    value = clean(name)
    if not value:
        return ""
    k = _key(value)
    return next((e for e in existing if _key(e) == k), value)
