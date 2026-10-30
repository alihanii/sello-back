"""Persian text helpers. Caption parsing and building are ported from platform-wrappers
(src/rubika/caption_parse.py and platform_hub/services/messenger_transfer.py)."""

import re

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_TO_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_PRICE_LINE = re.compile(r"^(?:💰\s*)?قیمت\s*[:：]\s*", re.IGNORECASE)
_INVENTORY_LINE = re.compile(r"^موجودی\s*[:：]\s*", re.IGNORECASE)
_CODE_LINE = re.compile(r"^کد(?:\s*محصول)?\s*[:：]\s*", re.IGNORECASE)
_SEPARATOR_LINE = re.compile(r"^-{3,}$")

MAX_CAPTION_CHARS = 1000


def normalize_fa(text: str) -> str:
    return str(text or "").replace("ي", "ی").replace("ك", "ک").replace("‌‌", "‌")


def to_en_digits(text: str) -> str:
    return str(text or "").translate(_PERSIAN_DIGITS)


def fa_number(value: int) -> str:
    return f"{int(value):,}".translate(_TO_FA_DIGITS).replace(",", "٬")


def format_price(price: int | float | None) -> str | None:
    try:
        value = int(price or 0)
    except (TypeError, ValueError):
        return None
    return f"{fa_number(value)} تومان" if value > 0 else None


def parse_number(text: str) -> int:
    normalized = to_en_digits(text)
    normalized = re.sub(r"[٬،,٬]", "", normalized)
    multiplier = 1
    if "میلیون" in normalized:
        multiplier = 1_000_000
    elif "هزار" in normalized:
        multiplier = 1_000
    normalized = re.sub(r"[^\d.]", "", normalized)
    if not normalized:
        return 0
    try:
        return max(int(float(normalized) * multiplier), 0)
    except ValueError:
        return 0


def _caption_lines(caption: str) -> list[str]:
    text = normalize_fa(caption).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return [ln for ln in lines if not _SEPARATOR_LINE.match(ln)]
    text = lines[0] if lines else text
    marker = r"(?:کد|قیمت|موجودی)\s*:"
    text = re.sub(rf"\s+(?={marker})", "\n", text)
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not _SEPARATOR_LINE.match(ln.strip())]


def parse_caption(caption: str) -> dict:
    """Cheap rule-based extraction. Returns `complete` False when an LLM pass should follow."""
    price = qty = 0
    code = ""
    body: list[str] = []
    for line in _caption_lines(caption):
        if _PRICE_LINE.match(line):
            price = parse_number(_PRICE_LINE.sub("", line))
        elif _INVENTORY_LINE.match(line):
            qty = parse_number(_INVENTORY_LINE.sub("", line))
        elif _CODE_LINE.match(line):
            code = to_en_digits(_CODE_LINE.sub("", line)).strip()
        else:
            body.append(line)
    title = body.pop(0) if body else ""
    hashtags = re.findall(r"#[\w؀-ۿ_]+", caption or "")
    return {
        "title": title[:300],
        "description": "\n".join(body).strip(),
        "price": price,
        "quantity": qty,
        "sku": code,
        "hashtags": hashtags,
        "complete": bool(title and price),
    }


def build_caption(
    *, title: str, description: str = "", price: int = 0, show_price: bool = False, signature: str = ""
) -> str:
    """Title, description and (only when the store opted in) the price. Room for the price
    line and signature is reserved first so a long description never silently drops them."""
    head = str(title or "").strip()
    body = str(description or "").strip()
    price_line = format_price(price) if show_price else None
    tail = f"\n\n💰 قیمت: {price_line}" if price_line else ""
    if signature.strip():
        tail += f"\n\n{signature.strip()}"
    budget = MAX_CAPTION_CHARS - len(head) - len(tail)
    if body and budget > 4:
        if len(body) + 2 > budget:
            body = body[: max(budget - 3, 0)].rstrip() + "…"
        parts = [head, body] if head else [body]
    else:
        parts = [head] if head else []
    return ("\n\n".join(p for p in parts if p) + tail).strip()[:MAX_CAPTION_CHARS]


def caption_for(product, store_settings: dict | None) -> str:
    """The text a channel posts: the AI-written post caption when there is one, else title + description."""
    st = store_settings or {}
    opts = {"price": product.price, "show_price": st.get("show_price_in_caption", True), "signature": st.get("caption_signature", "")}
    if (product.post_caption or "").strip():
        return build_caption(title="", description=product.post_caption, **opts)
    return build_caption(title=product.title, description=product.description, **opts)
