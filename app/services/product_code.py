"""Product code (SKU) defaults, rule-based, no AI.

A new product with an empty SKU gets the code written in its description ("کد: A120", "کد محصول ۳۴۵",
"SKU: X-9"); when the text has none, it gets the product's own id. The code is then written as the first
line of the description, so the posts and the chat show it."""

import re

from sqlalchemy import select

from app.models import Product
from app.services.text import normalize_fa, to_en_digits

# "کد", "کد محصول", "کد کالا", "code", "sku", "#کد_123"; the code itself must hold a digit or latin letter,
# so "کد تخفیف" or "کدهای" aren't read as codes.
_CODE = re.compile(
    r"(?<![\w‌])(?:کد(?:[\s_‌]*(?:محصول|کالا|مدل))?|sku|code|product\s*code)(?![^\W_]|‌)"
    r"[\s_]*[:：#=\-–]?[\s]*([A-Za-z0-9](?:[A-Za-z0-9\-_/.]{0,38}[A-Za-z0-9])?)",
    re.IGNORECASE,
)
CODE_LINE_PREFIX = "کد محصول:"
_CODE_LINE = re.compile(r"^\s*(?:کد(?:\s*(?:محصول|کالا))?|sku|code)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)


def find_code(*texts: str) -> str:
    """The first product code written in any of the texts, in latin digits, or ""."""
    for text in texts:
        m = _CODE.search(to_en_digits(normalize_fa(text or "")))
        if m:
            return m.group(1)[:80]
    return ""


def with_code_first(description: str, code: str) -> str:
    """The description with "کد محصول: <code>" as its first line, once. A code line already on top is replaced,
    and a line elsewhere that only states this same code is dropped so the code isn't written twice."""
    lines = (description or "").strip().splitlines()
    while lines and _CODE_LINE.match(to_en_digits(normalize_fa(lines[0]))):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    kept = []
    for line in lines:
        m = _CODE_LINE.match(to_en_digits(normalize_fa(line)))
        if not (m and m.group(1).lower() == code.lower()):
            kept.append(line)
    head = f"{CODE_LINE_PREFIX} {code}"
    return "\n".join([head, *kept]) if kept else head


async def _taken(db, store_id: int, code: str, product_id: int) -> bool:
    other = await db.scalar(select(Product.id).where(Product.store_id == store_id, Product.sku == code,
                                                     Product.id != product_id).limit(1))
    return other is not None


async def apply_default_code(db, product: Product, *extra_texts: str) -> None:
    """Call after the new product is flushed (it needs its id). Fills an empty SKU and puts it on top of the
    description. `extra_texts` are other things the seller wrote, e.g. the note sent with an AI request."""
    code = (product.sku or "").strip()
    if not code:
        found = find_code(product.description, *extra_texts)
        if found and not await _taken(db, product.store_id, found, product.id):
            code = found
        else:
            code = str(product.id)
            if await _taken(db, product.store_id, code, product.id):
                code = f"P{product.id}"
        product.sku = code
    product.description = with_code_first(product.description, code)
