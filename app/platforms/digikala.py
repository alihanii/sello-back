"""Digikala seller panel. HTTP calls ported from platform-wrappers
(digikala.api/app/services/digikala_seller_client.py and digikala_seller_products.py).
These are the seller PWA's internal endpoints, not a public API, so they can change."""

import io
import json
import re
import secrets
import string
from typing import Any
from urllib.parse import quote

import httpx

from app.core.security import decrypt_credentials
from app.models import Channel, Product, Store
from app.platforms.base import EDIT_PRODUCT, RECONNECT, CrawledItem, PlatformError, PublishResult
from app.services import ai
from app.services.images import load_bytes

BASE_URL = "https://seller.digikala.com"
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "x-web-optimize-response": "1",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/pwa/account/sign-in",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0 Safari/537.36",
}


def _digits(v: str) -> str:
    return re.sub(r"\D", "", v or "")


def login_username(phone: str) -> str:
    d = _digits(phone)
    return "0" + d[2:] if d.startswith("98") and len(d) == 12 else d


def verify_username(phone: str) -> str:
    d = _digits(phone)
    return d if d.startswith("98") else "98" + d.lstrip("0")


MISC_BRAND_ID = 719  # «متفرقه»


def _norm(v: str) -> str:
    return (v or "").strip().replace("ي", "ی").replace("ك", "ک").lower()

# Seller words Digikala doesn't use as category names, mapped to the ones it does.
_CATEGORY_WORDS = {
    "اورشرت": ["پیراهن مردانه", "پیراهن"], "اوورشرت": ["پیراهن مردانه", "پیراهن"], "اور": ["پیراهن"],
    "اورسایز": ["تی شرت"], "شومیز": ["شومیز و بلوز زنانه", "بلوز"], "هودی": ["سویشرت و هودی", "هودی"],
    "سویشرت": ["سویشرت و هودی"], "بافت": ["پلیور و بافت", "پلیور"], "پلیور": ["پلیور و بافت"],
    "تیشرت": ["تی شرت", "تی شرت و پولو شرت"], "تی‌شرت": ["تی شرت"], "پولوشرت": ["تی شرت و پولو شرت"],
    "کاپشن": ["کاپشن و پافر"], "پافر": ["کاپشن و پافر"], "شلوارک": ["شلوارک"], "اسلش": ["شلوار"],
    "جین": ["شلوار جین"], "کراپ": ["تاپ و کراپ"], "ست": ["ست لباس"],
    "دورس": ["سویشرت و هودی", "ست لباس"], "گرمکن": ["ست لباس ورزشی", "گرمکن"], "شلوار": ["شلوار"],
}


_STOP = {"و", "با", "طرح", "دار", "مدل", "برند", "جدید", "اصل", "کد", "رنگ"}


def _words(v: str) -> set[str]:
    return {w for w in re.split(r"[\s‌\-،,/>›()]+", _norm(v)) if len(w) > 1 and w not in _STOP}


def _overlap(product: Product, cat: dict) -> int:
    """Whole words the product's title/category hint shares with a Digikala category (with the word map's names)."""
    mine = _words(f"{product.title} {product.category_hint or ''}")
    mine |= {x for w in mine for name in _CATEGORY_WORDS.get(w, []) for x in _words(name)}
    return len(mine & (_words(cat.get("title") or "") | _words(str(cat.get("path") or ""))))


_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _from_text(field: dict, text: str) -> Any:
    """A value for a weight, size or material field written plainly in the product's title/description, else None."""
    t = text.translate(_FA_DIGITS)
    name = _norm(field["title"])
    if "وزن" in name:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(کیلو|kg|گرم|g\b)", t, re.I)
        if m:
            n = float(m.group(1).replace(",", "."))
            grams = n * 1000 if m.group(2).lower() in ("کیلو", "kg") else n
            return int(grams) if field["type"] == "number" else f"{int(grams)} گرم"
    if "ابعاد" in name or "اندازه" in name:
        m = re.search(r"\d+(?:\.\d+)?\s*[x×*]\s*\d+(?:\.\d+)?(?:\s*[x×*]\s*\d+(?:\.\d+)?)?(?:\s*(?:سانتی‌?متر|سانت|cm|میلی‌?متر|mm))?",
                      t, re.I)
        if m and field["type"] != "number":
            return m.group(0)
    if "جنس" in name and field["type"] not in ("number",):
        m = re.search(r"جنس(?:\s*(?:پارچه|بدنه))?\s*[:：]?\s*([^\s،,.\n]+)", t)
        if m:
            word = m.group(1)
            if field["options"]:
                hit = next((o["id"] for o in field["options"] if _norm(word) in _norm(o.get("text") or "")), None)
                return ([hit] if field["type"] == "multi_select" else hit) if hit is not None else None
            return word
    return None


def _product_id(payload: Any) -> Any:
    """The new product's id in a save answer: {data: {product_id}}, {data: {data: {product_id}}}, {data: {product: {id}}}."""
    for _ in range(4):
        if not isinstance(payload, dict):
            return None
        if payload.get("product_id"):
            return payload["product_id"]
        if isinstance(payload.get("product"), dict) and payload["product"].get("id"):
            return payload["product"]["id"]
        payload = payload.get("data")
    return None


def _ai_skipped(exc: Exception) -> str:
    return f"هوش مصنوعی جواب نداد ({getattr(exc, 'message_fa', '') or 'خطا'})؛ بدون آن ادامه داده شد."


def _category_items(payload: Any) -> list[dict]:
    """Leaf categories in a search answer, whatever its shape (data.items, data.categories, a list...)."""
    out: list[dict] = []

    def walk(v: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, dict):
            name = v.get("text") or v.get("title") or v.get("title_fa") or v.get("name")
            if v.get("id") and name and isinstance(name, str) and v.get("leaf", v.get("is_leaf", True)) is not False:
                out.append({"id": v["id"], "title": name, "path": v.get("path") or v.get("breadcrumb") or ""})
            for k, x in v.items():
                if isinstance(x, (dict, list)) and k not in ("path", "breadcrumb", "parent", "parents"):
                    walk(x, depth + 1)

    walk(payload.get("data") if isinstance(payload, dict) else payload)
    seen, uniq = set(), []
    for c in out:
        if c["id"] not in seen:
            seen.add(c["id"])
            uniq.append(c)
    return uniq



def _tokens(*texts: str) -> tuple[str, list[str]]:
    blob = _norm(" ".join(texts))
    return blob, [t for t in re.findall(r"[\w؀-ۿ]+", blob) if len(t) > 1]


def _pick_item(items: list, title: str, desc: str) -> Any:
    """Best bind option (division / product type) by word overlap with the product text, else the first."""
    blob, toks = _tokens(title, desc)
    best, best_score = None, 0
    for it in items:
        if not isinstance(it, dict):
            continue
        label = _norm(str(it.get("text") or it.get("title") or ""))
        value = it.get("value", it.get("id"))
        score = (20 if label and label in blob else 0) + sum(8 for t in toks if t in label)
        if value is not None and score > best_score:
            best, best_score = value, score
    if best is None and items and isinstance(items[0], dict):
        best = items[0].get("value", items[0].get("id"))
    return int(best) if best is not None and str(best).isdigit() else best


def _pick_mefa(mefa: dict, title: str, desc: str) -> Any:
    _, toks = _tokens(title, desc)
    best, best_score = None, -1
    for it in mefa.values():
        if not isinstance(it, dict):
            continue
        label = _norm(str(it.get("text") or ""))
        score = sum(6 for t in toks if t in label) + (2 if str(it.get("general_id") or "").startswith("272") else 0)
        if score > best_score:
            best, best_score = it.get("value"), score
    return best


def _package_specs(title: str, desc: str) -> dict[str, int]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*گرم", f"{title} {desc}")
    weight = int(float(m.group(1))) if m else 500
    return {"package_length": 10, "package_width": 10, "package_height": 10, "package_weight": max(weight, 50)}


def _attribute_fields(data: dict) -> list[dict[str, Any]]:
    fields = []
    groups = data.get("category_group_attributes") or {}
    for group in (groups.values() if isinstance(groups, dict) else groups):
        attrs = (group or {}).get("attributes") or {}
        for a in (attrs.values() if isinstance(attrs, dict) else attrs):
            if not isinstance(a, dict) or a.get("id") is None:
                continue
            raw = a.get("values") or {}
            options = [{"id": int(k) if str(k).isdigit() else k, "text": (v or {}).get("text", "")}
                       for k, v in (raw.items() if isinstance(raw, dict) else [])]
            kind = str(a.get("type") or "").lower()
            kind = ("multi_select" if kind in ("checkbox", "multiple", "multi_select") else
                    "select" if options else "number" if kind in ("number", "numeric", "integer") else "text")
            fields.append({"id": a["id"], "title": a.get("title") or "", "type": kind, "unit": a.get("unit") or "",
                           "required": bool(a.get("required")), "options": options})
    return fields


def _coerce(field: dict, value: Any) -> Any:
    if value in (None, "", []):
        return None
    opts = field["options"]

    def opt(v: Any) -> Any:
        for o in opts:
            if str(o["id"]) == str(v) or _norm(o["text"]) == _norm(str(v)):
                return o["id"]
        return None

    if field["type"] == "multi_select":
        picked = [x for x in (opt(v) for v in (value if isinstance(value, list) else [value])) if x is not None]
        return picked or None
    if field["type"] == "select":
        return opt(value[0] if isinstance(value, list) else value)
    if field["type"] == "number":
        m = re.search(r"\d+(?:\.\d+)?", str(value).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")))
        return (float(m.group()) if "." in m.group() else int(m.group())) if m else None
    return str(value)[:255]


# ---------- pictures: Digikala's image rules ----------
# Digikala seller academy: JPG, square, product centred on a pure white background, at least 600x600 and at most
# 2500x2500 pixels, under 5 MB (we keep well under), no text, logos, watermarks or frames.
DK_MIN_SIDE, DK_TARGET, DK_MAX_SIDE, DK_MAX_BYTES = 600, 1200, 2500, 3 * 1024 * 1024


def digikala_jpeg(data: bytes, fill: float = 0.94) -> bytes:
    """Square white-background JPEG within Digikala's size limits: the photo is fitted whole (no crop) and padded
    with white; small photos are scaled up to 1200x1200, large ones down to 2000 at most. `fill` is how much of the
    square the product takes."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        side = max(img.size)
        side = DK_TARGET if side < DK_TARGET else min(side, 2000)
        fitted = ImageOps.contain(img, (int(side * fill), int(side * fill)), Image.LANCZOS)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(fitted, ((side - fitted.width) // 2, (side - fitted.height) // 2))
        for quality in (90, 84, 76, 68):
            out = io.BytesIO()
            canvas.save(out, "JPEG", quality=quality, optimize=True)
            if out.tell() <= DK_MAX_BYTES:
                break
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise PlatformError("یکی از عکس‌ها برای دیجی‌کالا قابل تبدیل نبود.", "bad_image", EDIT_PRODUCT, str(exc)) from exc


def _image_ids(payload: Any) -> list:
    """Every image id in a temp_images answer, whatever its shape ({data: {id}}, {data: {data: {id}}}, items...)."""
    out: list = []

    def walk(v: Any, depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(v, dict):
            if v.get("id") is not None and any(k in v for k in ("url", "src", "image", "urls", "is_main", "main",
                                                                     "reference", "reference_id", "path", "name")):
                out.append(v["id"])
            for k, x in v.items():
                if isinstance(x, (dict, list)):
                    walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)

    walk(payload)
    return out


def _image_id(payload: Any) -> Any:
    data = payload.get("data") if isinstance(payload, dict) else None
    for d in (data, (data or {}).get("data") if isinstance(data, dict) else None,
              (data or {}).get("image") if isinstance(data, dict) else None,
              (data or {}).get("temp_image") if isinstance(data, dict) else None):
        if isinstance(d, dict) and d.get("id") is not None:
            return d["id"]
    ids = _image_ids(payload)
    return ids[0] if ids else None


# ---------- after publishing: review state, variant (price + stock), orders ----------

def review_state(code: str) -> str:
    """Digikala's moderation code ("waiting_for_confirm", "rejected"...) -> in_review | needs_edit | rejected | approved."""
    c = (code or "").lower()
    if not c:
        return ""
    if "reject" in c or "denied" in c or "decline" in c:
        return "rejected"
    if "edit" in c or "correct" in c or "revis" in c or "need" in c:
        return "needs_edit"
    if "wait" in c or "pending" in c or "review" in c or "draft" in c or "check" in c or "new" == c:
        return "in_review"
    if "approv" in c or "confirm" in c or "accept" in c or "active" in c or "publish" in c:
        return "approved"
    return "in_review"


REVIEW_FA = {"in_review": "در انتظار تأیید دیجی‌کالا", "needs_edit": "نیاز به اصلاح", "rejected": "رد شده",
             "approved": "تأیید شده", "live": "فعال در دیجی‌کالا (قیمت و موجودی ثبت شد)"}

# Our order statuses for Digikala's: an order on Digikala is already paid by the customer, so anything that isn't
# cancelled or returned counts as confirmed; Digikala's own label is kept beside it.
ORDER_FA = {"new": "سفارش جدید", "pending": "در انتظار پردازش", "processing": "در حال پردازش",
            "ready_to_ship": "آمادهٔ ارسال", "shipped": "ارسال شده", "delivered": "تحویل شده",
            "cancelled": "لغو شده", "canceled": "لغو شده", "returned": "مرجوع شده", "failed_delivery": "تحویل ناموفق"}


def order_status(code: str) -> str:
    c = (code or "").lower()
    return "cancelled" if any(w in c for w in ("cancel", "return", "reject", "refund")) else "confirmed"


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        cur: Any = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur not in (None, "", [], {}):
            return cur
    return None


def _text(v: Any) -> str:
    if isinstance(v, dict):
        v = _first(v, "title", "text", "label", "name", "fa", "translate", "value")
    return str(v).strip() if v not in (None, "") else ""


def _rows(data: Any) -> list[dict]:
    """The list of rows in a panel response, wherever it sits (data.items, data.orders, data.data...)."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if not isinstance(data, dict):
        return []
    for k in ("items", "orders", "shipments", "results", "rows", "list", "data"):
        v = data.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
        if isinstance(v, dict):
            inner = _rows(v)
            if inner:
                return inner
    return []


def _toman(v: Any) -> int:
    """Digikala's panel speaks rial; Sello stores toman."""
    try:
        return int(float(str(v).replace(",", ""))) // 10
    except (TypeError, ValueError):
        return 0


def normalize_order(row: dict) -> dict[str, Any] | None:
    oid = _first(row, "order_item_id", "shipment_id", "id", "order_id", "order.id")
    if oid is None:
        return None
    status = _first(row, "status.code", "status.value", "status", "order_status", "shipment_status", "state")
    status_code = status if isinstance(status, str) else _text(_first(row, "status.code", "status.name")) or str(status or "")
    label = _text(_first(row, "status.title", "status.text", "status.label", "status_label", "status_title",
                         "status_fa", "status_translate")) or ORDER_FA.get(status_code.lower(), status_code)
    raw_items = _first(row, "items", "order_items", "variants", "products") or [row]
    items = []
    for it in raw_items if isinstance(raw_items, list) else [raw_items]:
        if not isinstance(it, dict):
            continue
        pid = _first(it, "product_id", "product.id", "variant.product_id")
        price = _first(it, "selling_price", "price", "unit_price", "variant.price.selling_price", "price.selling_price")
        items.append({"remote_product_id": int(pid) if str(pid or "").isdigit() else None,
                      "variant_id": _first(it, "variant_id", "variant.id", "dkpc"),
                      "title": _text(_first(it, "product_title", "title_fa", "title", "product.title_fa", "variant.title_fa",
                                            "product.title")) or "کالای دیجی‌کالا",
                      "qty": int(_first(it, "quantity", "qty", "count") or 1),
                      "unit_price": _toman(price),
                      "image_url": _text(_first(it, "image_src", "image", "product.image_src", "variant.image_src"))})
    created = _first(row, "created_at", "order_created_at", "order.created_at", "date", "created_date")
    customer = _first(row, "customer", "customer_info", "receiver", "address") or {}
    return {"id": str(oid), "code": str(_first(row, "order_code", "code", "order_id", "order.code") or oid),
            "status_code": status_code[:40], "status_fa": label[:120], "status": order_status(status_code),
            "created_at": str(created) if created else "",
            "customer_name": _text(_first(customer if isinstance(customer, dict) else {}, "full_name", "name")
                                   or _first(row, "customer_name", "receiver_name")),
            "city": _text(_first(customer if isinstance(customer, dict) else {}, "city", "city_name")
                          or _first(row, "city", "city_name", "destination_city")),
            "items": items, "raw": row}


def variant_body(form: dict, product: Product, dk_product_id: int) -> dict[str, Any]:
    """The variant form's defaults (first warranty, first theme value such as the color) plus our price and stock."""
    form = form if isinstance(form, dict) else {}

    def first_id(*keys: str) -> Any:
        for k in keys:
            v = form.get(k)
            items = list(v.values()) if isinstance(v, dict) else v if isinstance(v, list) else []
            for it in items:
                if isinstance(it, dict) and it.get("id", it.get("value")) is not None:
                    return it.get("id", it.get("value"))
        return None

    price = int(product.price) * 10
    body: dict[str, Any] = {
        "product_id": dk_product_id,
        "price": price, "selling_price": price, "price_list": max(price, int(product.compare_at_price or 0) * 10),
        "seller_stock": max(int(product.stock_qty or 0), 0), "stock": max(int(product.stock_qty or 0), 0),
        "lead_time": int(form.get("default_lead_time") or 2), "maximum_per_order": int(form.get("maximum_per_order") or 5),
        "shipping_type": form.get("default_shipping_type") or "digikala",
        "warranty_id": first_id("warranties", "warranty"),
        "theme_value_id": first_id("theme_values", "colors", "sizes", "themes"),
        "seller_code": (product.sku or f"SL-{product.id}")[:40],
    }
    return {k: v for k, v in body.items() if v is not None}


ORDER_PATHS = ("/api/v2/ship-by-seller-orders", "/api/v2/orders/")


class DigikalaAdapter:
    platform = "digikala"
    label_fa = "دیجی‌کالا"

    async def _request(self, method: str, path: str, *, token: str | None = None, body: dict | None = None,
                       referer: str | None = None, files: dict | None = None,
                       data: dict | None = None) -> dict[str, Any]:
        headers = dict(DEFAULT_HEADERS)
        if files:
            headers.pop("Content-Type")
        if referer:
            headers["Referer"] = referer
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            # Shorter than nginx's 60 s, so a slow or silent Digikala answers the seller with a clear message.
            async with httpx.AsyncClient(timeout=httpx.Timeout(25, connect=10), follow_redirects=True,
                                         headers=headers) as client:
                if files:
                    resp = await client.request(method, f"{BASE_URL}{path}", files=files, data=data)
                else:
                    resp = await client.request(method, f"{BASE_URL}{path}", json=body)
        except httpx.TimeoutException as exc:
            raise PlatformError("پنل فروشندگان دیجی‌کالا جواب نداد (زمان انتظار تمام شد)؛ احتمالاً دیجی‌کالا IP سرور را "
                                "مسدود کرده یا کند است. چند دقیقه بعد دوباره امتحان کن.", "timeout",
                                detail=f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise PlatformError("ارتباط با پنل فروشندگان دیجی‌کالا برقرار نشد.", "network",
                                detail=f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 401:
            raise PlatformError("نشست دیجی‌کالا منقضی شده؛ دوباره با کد تأیید وارد شوید.", "expired", RECONNECT,
                                resp.text[:500])
        if resp.status_code >= 400:
            msg = self._extract_message(resp)
            blocked = resp.status_code in (403, 429) and "json" not in resp.headers.get("content-type", "")
            text = (f"دیجی‌کالا خطا داد: {msg}" if msg else
                    f"دیجی‌کالا درخواست سرور را مسدود کرد (کد {resp.status_code})؛ احتمالاً به‌خاطر IP سرور یا کپچا." if blocked else
                    f"دیجی‌کالا درخواست را رد کرد (کد {resp.status_code}).")
            raise PlatformError(text, f"http_{resp.status_code}", detail=resp.text[:1000])
        try:
            return resp.json()
        except ValueError:
            raise PlatformError("دیجی‌کالا پاسخ نامعتبر داد.", "bad_response", detail=resp.text[:300])

    @staticmethod
    def _extract_message(resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except ValueError:
            return ""
        if not isinstance(data, dict):
            return ""
        parts = []
        for key in ("message", "error"):
            if isinstance(data.get(key), str) and data[key].strip():
                parts.append(data[key].strip())
        errors = data.get("errors")
        if isinstance(errors, dict):
            for v in errors.values():
                parts.append("؛ ".join(map(str, v)) if isinstance(v, list) else str(v))
        elif isinstance(errors, list):
            parts.extend(map(str, errors))
        return " — ".join(parts)[:400]

    # ---------- connect: two calls (send code, then verify) ----------

    async def send_code(self, ctx, phone: str) -> dict[str, Any]:
        username = login_username(phone)
        async with ctx.step("login", "شروع ورود به پنل فروشنده", request={"username": username}) as s:
            s.response = await self._request("POST", "/api/v2/login", body={"username": username, "captcha_token": ""})
        async with ctx.step("send_otp", "ارسال کد تأیید دیجی‌کالا", request={"username": username}) as s:
            payload = await self._request("POST", "/api/v2/otp/send", body={"username": username, "captcha_token": ""})
            s.response = payload
        challenge = (payload.get("data") or {}).get("challenge_token")
        if not challenge:
            raise PlatformError("دیجی‌کالا کد تأیید را ارسال نکرد؛ چند دقیقه بعد دوباره تلاش کنید.", "no_challenge")
        return {"challenge_token": challenge, "phone": username}

    async def connect(self, ctx, channel: Channel, data: dict[str, Any]) -> dict[str, Any]:
        pending = channel.config.get("pending") or {}
        if not pending.get("challenge_token"):
            raise PlatformError("اول کد تأیید را درخواست کنید.", "no_pending_login")
        async with ctx.step("verify_otp", "تأیید کد", request={"username": verify_username(pending["phone"])}) as s:
            payload = await self._request(
                "POST", "/api/v2/otp/verify", referer=f"{BASE_URL}/pwa/account/otp",
                body={"otp": data["code"], "username": verify_username(pending["phone"]),
                      "challenge_token": pending["challenge_token"]})
            s.response = {"ok": True}
        token = ((payload.get("data") or {}).get("token") or {})
        if not token.get("access_token"):
            raise PlatformError("کد تأیید اشتباه یا منقضی است.", "bad_otp", RECONNECT)
        async with ctx.step("profile", "دریافت مشخصات فروشنده") as s:
            profile = (await self._request("GET", "/api/v2/profile?mini=true", token=token["access_token"],
                                           referer=f"{BASE_URL}/pwa/profile")).get("data") or {}
            s.response = {k: profile.get(k) for k in ("seller_id", "business_name", "name") if k in profile}
        return {
            "credentials": {"access_token": token["access_token"], "refresh_token": token.get("refresh_token")},
            "config": {"phone": pending["phone"], "seller_id": profile.get("seller_id"), "pending": None},
            "display_name": profile.get("business_name") or profile.get("name") or f"دیجی‌کالا {pending['phone']}",
            "external_account_id": str(profile.get("seller_id") or pending["phone"]),
        }

    def _token(self, channel: Channel) -> str:
        token = decrypt_credentials(channel.credentials_encrypted).get("access_token")
        if not token:
            raise PlatformError("حساب دیجی‌کالا وصل نیست.", "missing_token", RECONNECT)
        return token

    async def health(self, ctx, channel: Channel) -> None:
        async with ctx.step("profile", "بررسی نشست دیجی‌کالا") as s:
            data = await self._request("GET", "/api/v2/profile?mini=true", token=self._token(channel),
                                       referer=f"{BASE_URL}/pwa/profile")
            s.response = {"seller_id": (data.get("data") or {}).get("seller_id")}

    def validate(self, product: Product) -> list[str]:
        errors = []
        if len(product.title or "") < 5:
            errors.append("عنوان محصول برای دیجی‌کالا خیلی کوتاه است.")
        if not product.images:
            errors.append("دیجی‌کالا حداقل یک عکس لازم دارد.")
        return errors

    # ---------- publish: the seller panel's product-creation wizard, step by step ----------
    # Ported from platform-wrappers digikala_seller_product_worker: category → draft → attributes →
    # title → images → save → submit for review. AI decisions go through our admin-routed providers.

    @staticmethod
    async def ensure_enabled(db) -> None:
        from app.services.integrations import digikala_app

        if db is not None and not (await digikala_app(db))["enabled"]:
            raise PlatformError("دیجی‌کالا فعلاً توسط ادمین سایت خاموش شده است.", "platform_off")

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
        await self.ensure_enabled(ctx.db)
        if existing_remote_id:
            raise PlatformError("این محصول قبلاً در دیجی‌کالا ثبت شده؛ ویرایش آن فعلاً از پنل دیجی‌کالا انجام می‌شود.",
                                "already_published", {"type": "info", "label": "ویرایش در پنل دیجی‌کالا"})
        token = self._token(channel)
        title, desc = product.title.strip(), (product.description or "").strip()

        category, bind = await self._pick_category(ctx, token, product)
        cat_id = int(category["id"])
        ref = f"{BASE_URL}/pwa/product/create?categoryId={cat_id}"
        package = _package_specs(title, desc)

        draft_id = await self._create_draft(ctx, token, product, cat_id, bind, package, ref)
        ref = f"{ref}&draftProductId={draft_id}"
        await self._fill_attributes(ctx, token, product, cat_id, draft_id, package, ref)

        copy = await self._ai_copy(ctx, product, desc)
        async with ctx.step("title", "ثبت عنوان و توضیح", request={"title_fa": title}) as s:
            auto = await self._request("GET", f"/api/v2/product-creation/{draft_id}/auto-title", token=token, referer=ref)
            text = desc if len(desc) >= 150 else copy.get("description") or ""
            body = {"draft_product_id": draft_id, "title_fa": title[:255], "title_en": copy.get("title_en", "")[:255],
                    "description": text if len(text) >= 150 else None,
                    "advantages": copy.get("advantages", []), "disadvantages": copy.get("disadvantages", [])}
            s.response = {"auto_title": (auto.get("data") or {}).get("title_fa"),
                          "save": await self._request("POST", "/api/v2/product-creation/auto-title/save",
                                                      token=token, body=body, referer=ref)}

        urls = [img.url for img in product.images[:8]]
        main = await self._main_image(ctx, product, desc)
        if main:
            urls = [main] + urls[1:]
        # Digikala wants at least two pictures: with one, a second copy of it goes, framed a little wider so it
        # isn't the same file twice.
        shots = [(u, 0.94) for u in urls]
        if len(shots) == 1:
            shots.append((shots[0][0], 0.8))
        async with ctx.step("images", f"آپلود {len(shots)} عکس با استاندارد دیجی‌کالا") as s:
            ids, sizes = [], []
            for i, (url, fill) in enumerate(shots):
                data = digikala_jpeg(await load_bytes(url), fill)
                sizes.append(len(data))
                up = await self._request("POST", "/api/v2/product-creation/temp_images", token=token, referer=ref,
                                         files={"file": (f"product_{draft_id}_{i}.jpg", data, "image/jpeg")},
                                         data={"reference": "draft_product", "reference_id": str(draft_id)})
                image_id = _image_id(up)
                if not image_id:  # the upload answer's shape changes; the draft's own list is the source of truth
                    listed = await self._request("GET", f"/api/v2/product-creation/temp_images/draft_product/{draft_id}",
                                                 token=token, referer=ref)
                    known = [x for x in (_image_ids(listed)) if x not in ids]
                    image_id = known[-1] if known else None
                if not image_id:
                    raise PlatformError("دیجی‌کالا عکس را نپذیرفت: " + (self._message(up) or "پاسخ بدون شناسهٔ عکس"),
                                        "image_rejected", EDIT_PRODUCT, up)
                ids.append(image_id)
                if i == 0:
                    await self._request("PATCH", f"/api/v2/product-creation/temp_images/{image_id}/main",
                                        token=token, body=None, referer=ref)
            s.response = {"image_ids": ids, "bytes": sizes, "ai_main": bool(main), "copied_first": len(urls) == 1}

        async with ctx.step("save", "ثبت نهایی محصول") as s:
            saved = await self._request("POST", "/api/v2/product-creation/save", token=token, referer=ref,
                                        body={"category_id": cat_id, "draft_product_id": str(draft_id),
                                              "only_b2b": False, "use_temp_images": True})
            data = saved.get("data") or {}
            product_id = _product_id(saved)
            s.response = data
        if not product_id:
            raise PlatformError("دیجی‌کالا محصول را ذخیره نکرد؛ جزئیات در لاگ است.", "save_failed", detail=saved)

        async with ctx.step("submit", "ارسال برای بررسی دیجی‌کالا", request={"product_id": product_id}) as s:
            try:
                s.response = await self._request("POST", f"/api/v2/product-edit/{product_id}/publish",
                                                 token=token, body={}, referer=ref)
            except PlatformError as exc:
                # Saved products already sit in Digikala's review queue; an explicit submit is best effort.
                s.response = {"skipped": exc.message_fa}

        return PublishResult(
            f"DKP-{product_id}", f"https://www.digikala.com/product/dkp-{product_id}/",
            {"draft_product_id": draft_id, "category_id": cat_id, "category": category.get("title"),
             "note": "محصول در صف بررسی دیجی‌کالاست؛ پس از تأیید، قیمت و موجودی را در پنل دیجی‌کالا ثبت کنید."})

    @staticmethod
    def _message(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("message") or payload.get("error") or "")[:200]

    async def _main_image(self, ctx, product: Product, desc: str) -> str:
        """The main picture made the way «ساخت محصول هوشمند» makes it (Digikala's main-image rules: the product alone,
        centred, white background, no text), from the seller's first photo. Kept in source_ref so a retry reuses it.
        Runs whenever an image key is set in admin (the plan's AI-pictures switch doesn't apply here); empty when there is
        none or the model gives nothing: the seller's photo goes instead, made square on white."""
        if not product.images:
            return ""
        first = product.images[0]
        ref = dict(getattr(product, "source_ref", None) or {})
        cached = ref.get("digikala_main_image")
        if cached and cached.get("from") == first.url:
            return cached["url"]
        origin = getattr(first, "origin_url", "") or ""
        if origin.startswith("/media/") and origin != first.url:
            return ""  # already the cleaned main picture from «ساخت محصول هوشمند» (the seller's photo is its origin)
        if not await ai.ai_available("image_edit"):
            async with ctx.step("image_ai_skipped", "عکس اصلی با AI ساخته نشد؛ عکس خود محصول استفاده می‌شود") as s:
                s.response = {"reason": "در ادمین برای «ساخت عکس» (image_edit) کلید و مدل تنظیم نشده."}
            return ""
        from app.services.plans import MARKETPLACE_MAIN_IMAGE
        from app.services.product_images import make_photo

        allow = MARKETPLACE_MAIN_IMAGE.set(True)
        try:
            url = await make_photo(ctx, store_id=product.store_id, title=product.title,
                                   note=(product.description or desc or "")[:400],
                                   category="general", image_url=first.url)
        except Exception as exc:  # noqa: BLE001 - plan without AI pictures, model refused...: the seller's photo goes
            async with ctx.step("image_ai_skipped", "عکس AI ساخته نشد؛ عکس خود محصول استفاده می‌شود") as s:
                s.response = {"reason": getattr(exc, "message_fa", None) or str(exc)[:300]}
            return ""
        finally:
            MARKETPLACE_MAIN_IMAGE.reset(allow)
        ref["digikala_main_image"] = {"from": first.url, "url": url}
        product.source_ref = ref
        if getattr(ctx, "db", None) is not None and getattr(product, "id", None):
            # The same picture becomes the product's own main picture (the seller's photo stays after it), marked as
            # cleaned the way «ساخت محصول هوشمند» marks it, so it isn't made again.
            from app.services.jobs import _set_main_image

            await _set_main_image(ctx.db, product.id, url, first.url)
        return url

    async def _ai_copy(self, ctx, product: Product, desc: str) -> dict[str, Any]:
        """Description (Digikala wants 150+ characters), English title, advantages and disadvantages, written by AI from
        the product's own text. Empty when AI is off; the listing then goes without them."""
        if not await ai.ai_available("field_fill"):
            return {}
        try:
            async with ctx.step("copy_ai", "نوشتن توضیح، عنوان انگلیسی و نقاط قوت با هوش مصنوعی") as s:
                res = await ai.generate_json(
                    task="field_fill", store_id=product.store_id,
                    system=("You write a Digikala product listing in Persian from the seller's text. Use only facts in the "
                            "text; no prices, phone numbers, links, emojis or claims like 'best'. Reply JSON {"
                            "\"description\": 2-4 Persian sentences, 180-700 characters, "
                            "\"title_en\": short English product title, "
                            "\"advantages\": up to 3 short Persian points, \"disadvantages\": up to 1 short honest point or []}."),
                    prompt=json.dumps({"title": product.title, "description": desc[:1500], "brand": product.brand,
                                       "category": product.category_hint, "attributes": product.attributes},
                                      ensure_ascii=False))
                res = res if isinstance(res, dict) else {}
                out = {"description": str(res.get("description") or "").strip()[:2000],
                       "title_en": re.sub(r"[^\x20-\x7E]", "", str(res.get("title_en") or "")).strip(),
                       "advantages": [str(a).strip()[:100] for a in (res.get("advantages") or []) if str(a).strip()][:3],
                       "disadvantages": [str(a).strip()[:100] for a in (res.get("disadvantages") or []) if str(a).strip()][:1]}
                s.response = out
                return out
        except Exception:  # noqa: BLE001 - copy is optional; the listing still goes with the seller's own text
            return {}

    async def _search_categories(self, token: str, kw: str) -> tuple[list[dict], Any]:
        q = httpx.QueryParams({"keyword": kw})
        data = await self._request("GET", f"/api/v2/product-creation/search/category/v2/{quote(kw)}?{q}",
                                   token=token, referer=f"{BASE_URL}/pwa/product/create")
        return _category_items(data), data

    async def _category_terms(self, ctx, product: Product) -> list[str]:
        """What a Digikala seller would type in the category box for this product (e.g. «اورشرت» → «پیراهن مردانه»),
        written by AI. Empty when AI is off."""
        if not await ai.ai_available("field_fill"):
            return []
        try:
            res = await ai.generate_json(
                task="field_fill", store_id=product.store_id,
                system=("Digikala (Iranian marketplace) category search. Give the Persian names of the Digikala leaf "
                        "categories this product belongs in, the way Digikala names them (e.g. «پیراهن مردانه», "
                        "«سویشرت و هودی مردانه», «تی شرت و پولو شرت مردانه», «کیف دستی زنانه»), most likely first, then "
                        "1-2 shorter general words (e.g. «پیراهن»). No brand names. Reply JSON {\"terms\": [up to 5]}."),
                prompt=json.dumps({"title": product.title, "category": product.category_hint,
                                   "description": (product.description or "")[:400]}, ensure_ascii=False))
            terms = res.get("terms") if isinstance(res, dict) else None
            return [str(t).strip()[:60] for t in (terms or []) if str(t).strip()][:5]
        except Exception:  # noqa: BLE001
            return []

    async def _pick_category(self, ctx, token: str, product: Product) -> tuple[dict, dict]:
        title = re.sub(r"\s+(برند|مدل|طرح)\s+.*$", "", product.title).strip()  # «اورشرت برند X مدل Y» → «اورشرت»
        hint = (product.category_hint or "").replace("›", ">").replace("/", ">")
        keywords = [k.strip() for k in (hint.split(">")[-1], hint, title, " ".join(product.title.split()[:2]),
                                         product.title.split()[0]) if k and k.strip()]
        keywords += [w for t in _norm(product.title).split() + _norm(hint).split() for w in _CATEGORY_WORDS.get(t, [])]
        candidates: list[dict] = []
        tried: dict[str, Any] = {}

        async def search(kws: list[str]) -> None:
            for kw in dict.fromkeys(kws):
                if kw in tried or len(candidates) >= 5:
                    continue
                items, raw = await self._search_categories(token, kw)
                tried[kw] = len(items) if items else json.dumps(raw, ensure_ascii=False)[:300]
                for it in items:
                    if all(c["id"] != it["id"] for c in candidates):
                        candidates.append(it)

        async with ctx.step("category_search", "جستجوی دستهٔ مناسب در دیجی‌کالا", request={"keywords": keywords}) as s:
            await search(keywords)
            if len(candidates) < 2:  # the seller's words aren't Digikala's: ask AI what Digikala calls it
                await search(await self._category_terms(ctx, product))
            s.response = {"candidates": candidates[:10], "searched": tried}
        if not candidates:
            raise PlatformError("دستهٔ مناسبی در دیجی‌کالا برای این محصول پیدا نشد؛ «دسته» را در مشخصات محصول دقیق‌تر بنویسید "
                                "(مثلاً «پیراهن مردانه»).", "no_category", EDIT_PRODUCT, tried)

        # Without AI (off, or its key out of quota) the candidates closest to the product's own words go first.
        ordered = sorted(candidates[:10], key=lambda c: -_overlap(product, c))
        best = None
        if len(ordered) > 1 and await ai.ai_available("field_fill"):
            async with ctx.step("category_ai", "انتخاب دسته با هوش مصنوعی") as s:
                try:
                    res = await ai.generate_json(
                        task="field_fill", store_id=product.store_id,
                        system="You pick the best Digikala leaf category for a product. Reply JSON {\"id\": <id>}.",
                        prompt=json.dumps({"product": {"title": product.title, "category_hint": product.category_hint,
                                                       "description": (product.description or "")[:500]},
                                           "categories": ordered}, ensure_ascii=False))
                except Exception as exc:  # noqa: BLE001 - publishing goes on with the word match
                    res = {}
                    s.message_fa = _ai_skipped(exc)
                best = next((c for c in ordered if isinstance(res, dict) and str(c["id"]) == str(res.get("id"))), None)
                if best:
                    ordered = [best] + [c for c in ordered if c is not best]
                s.response = {"picked": best or ordered[0], "by": "ai" if best else "words"}
        if not best:
            # No AI pick: only categories sharing a word with the product (e.g. «ست» must not land in «استند و پایه
            # خنک‌کننده»); a guess would publish the product in the wrong place.
            sure = [c for c in ordered if _overlap(product, c)]
            if not sure:
                raise PlatformError("دستهٔ دیجی‌کالا بدون هوش مصنوعی مطمئن پیدا نشد؛ «دسته» را در مشخصات محصول با اسم "
                                    "دیجی‌کالایی بنویسید (مثلاً «ست لباس مردانه») یا اعتبار کلید AI را شارژ کنید. نزدیک‌ترین‌ها: "
                                    + "، ".join(c["title"] for c in ordered[:4]), "no_category", EDIT_PRODUCT)
            ordered = sure
        restricted = []
        for cat in ordered[:4]:
            async with ctx.step(f"category_{cat['id']}", f"بررسی مجوز فروش در «{cat['title']}»") as s:
                data = (await self._request("GET", f"/api/v2/product-creation/category/{cat['id']}/validation",
                                            token=token, referer=f"{BASE_URL}/pwa/product/create")).get("data") or {}
                s.response = {"is_valid": data.get("is_valid"), "errors": data.get("errors")}
            if data.get("is_valid", True) is not False:
                return cat, data.get("bind") or {}
            restricted.append(cat["title"])
        raise PlatformError(f"حساب فروشندهٔ شما اجازهٔ فروش در این دسته‌ها را ندارد: {'، '.join(restricted)}",
                            "category_restricted", {"type": "help", "label": "درخواست دسترسی در پنل دیجی‌کالا"})

    async def _create_draft(self, ctx, token: str, product: Product, cat_id: int, bind: dict,
                            package: dict, ref: str) -> int:
        title, desc = product.title, product.description or ""
        brands = [b for b in bind.get("brands") or [] if isinstance(b, dict) and b.get("id") is not None]
        want = _norm(product.brand)
        brand = next((b for b in brands if want and _norm(b.get("title_fa") or b.get("title_en") or "") == want), None)
        misc = next((b for b in brands if b["id"] == MISC_BRAND_ID or "متفرقه" in _norm(b.get("title_fa") or "")), None)
        brand_id = (brand or misc or {"id": MISC_BRAND_ID})["id"]
        model_match = re.search(r"مدل\s+([\w\-]+)", title)
        model = (model_match.group(1) if model_match else product.sku or
                 "SL-" + "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6)))[:20]
        body: dict[str, Any] = {
            "only_cf_fields": {"status": "marketable", "platforms": ["digikala"], "other_titles": []},
            "fake": False, "is_iranian": True, "brand_id": brand_id, "model": model,
            "division_id": _pick_item(bind.get("divisions") or [], title, desc),
            "general_mefa_id": _pick_mefa(bind.get("general_mefa") or {}, title, desc),
            "category_id": cat_id, "fake_reasons": [], "exclusive_mefa_id": None, **package,
        }
        if bind.get("category_product_types"):
            body["product_type_ids"] = [_pick_item(bind["category_product_types"], title, desc)]

        for attempt in range(2):
            async with ctx.step("draft" if not attempt else "draft_retry", "ساخت پیش‌نویس محصول",
                               request={k: body[k] for k in ("brand_id", "model", "category_id", "division_id")}) as s:
                try:
                    data = (await self._request("POST", "/api/v2/product-creation/product/detail/validation",
                                                token=token, body=body, referer=ref)).get("data") or {}
                except PlatformError as exc:
                    blob = json.dumps(exc.detail, ensure_ascii=False) if exc.detail else exc.message_fa
                    if attempt == 0 and brand and ("error_limitation" in blob or ("برند" in blob and "محدود" in blob)):
                        body["brand_id"] = (misc or {"id": MISC_BRAND_ID})["id"]
                        s.response = {"retry": "brand_limited_use_misc"}
                        continue
                    raise
                s.response = data
            draft_id = data.get("draft_product_id") or data.get("id")
            if draft_id:
                return int(draft_id)
            break
        raise PlatformError("دیجی‌کالا پیش‌نویس محصول را نساخت.", "draft_failed", detail=data)

    async def _fill_attributes(self, ctx, token: str, product: Product, cat_id: int, draft_id: int,
                               package: dict, ref: str) -> None:
        async with ctx.step("attributes_schema", "دریافت ویژگی‌های دسته") as s:
            data = (await self._request("GET", f"/api/v2/product-creation/attributes/{cat_id}?categoryId={cat_id}",
                                        token=token, referer=ref)).get("data") or {}
            fields = _attribute_fields(data)
            s.response = {"fields": len(fields), "required": sum(f["required"] for f in fields)}
        values: dict[str, Any] = {}
        mine = {_norm(k): v for k, v in (product.attributes or {}).items()}
        text = f"{product.title}\n{product.description or ''}"
        for f in fields:
            v = mine.get(_norm(f["title"]))
            if v in (None, ""):
                v = _from_text(f, text)  # «وزن ۴۵۰ گرم», «ابعاد ۳۰×۲۰», «جنس: پنبه» in the seller's own text
            if v not in (None, ""):
                values[str(f["id"])] = v
        # Every field still empty goes to the AI: required ones must get a value, optional ones only when the product
        # text (or the picture's obvious facts) says so; the more fields Digikala gets, the fewer review rejections.
        missing = [f for f in fields if str(f["id"]) not in values]
        missing.sort(key=lambda f: not f["required"])
        missing = missing[:60]
        if missing and await ai.ai_available("field_fill"):
            n_req = sum(f["required"] for f in missing)
            async with ctx.step("attributes_ai", f"پر کردن {len(missing)} ویژگی ({n_req} اجباری) با هوش مصنوعی") as s:
                try:
                    res = await ai.generate_json(
                        task="field_fill", store_id=product.store_id,
                        system=("Fill Digikala product attributes from the product text. For select fields answer with one "
                                "option id, for multi_select a list of option ids, for number a number, else short Persian "
                                "text. Fields with required=true must always get a value: when the text doesn't say, choose "
                                "the most neutral/likely option. For optional fields give a value only when the product text "
                                "makes it clear, else null. Never invent specs that contradict the text. "
                                "Reply JSON {\"<field id>\": value}."),
                        prompt=json.dumps({"product": {"title": product.title, "description": (product.description or "")[:1500],
                                                       "brand": product.brand, "category": product.category_hint,
                                                       "attributes": product.attributes},
                                           "fields": [{k: f[k] for k in ("id", "title", "type", "unit", "required", "options")}
                                                      for f in missing]},
                                          ensure_ascii=False))
                except Exception as exc:  # noqa: BLE001 - required fields still get the neutral first option below
                    res = {}
                    s.message_fa = _ai_skipped(exc)
                res = res if isinstance(res, dict) else {}
                values.update({str(k): v for k, v in res.items() if v not in (None, "", [])})
                s.response = res
        attrs = []
        for f in fields:
            v = _coerce(f, values.get(str(f["id"])))
            if v is None and f["required"] and f["options"]:
                v = [f["options"][0]["id"]] if f["type"] == "multi_select" else f["options"][0]["id"]
            if v is not None:
                attrs.append({"id": f["id"], "value": v})
        still = [f["title"] for f in fields if f["required"] and all(a["id"] != f["id"] for a in attrs)]
        if still:
            raise PlatformError(f"این ویژگی‌های اجباری دیجی‌کالا پر نشده: {'، '.join(still[:6])}. آن‌ها را در مشخصات محصول اضافه کنید.",
                                "missing_attributes", EDIT_PRODUCT)
        async with ctx.step("attributes", f"ثبت {len(attrs)} ویژگی", request={"attributes": attrs}) as s:
            s.response = await self._request("POST", "/api/v2/product-creation/attributes", token=token, referer=ref, body={
                "attributes": attrs, "category_id": cat_id, "draft_product_id": draft_id,
                "length": package["package_length"], "height": package["package_height"],
                "width": package["package_width"], "weight": package["package_weight"]})

    # ---------- after publishing ----------

    async def review_statuses(self, ctx, channel: Channel, product_ids: set[int], max_pages: int = 20) -> dict[int, dict]:
        """{dkp id: {code, label, state, variants_count}} for the given products, from the seller's products list."""
        token = self._token(channel)
        labels: dict[str, str] = {}
        try:
            groups = (await self._request("GET", "/api/v2/products/filters/seller", token=token,
                                          referer=f"{BASE_URL}/pwa/product-management")).get("data") or []
            for group in groups if isinstance(groups, list) else []:
                for field in group if isinstance(group, list) else [group]:
                    if isinstance(field, dict) and field.get("name") == "moderation_status" and isinstance(field.get("data"), dict):
                        labels = {str(k): str(v) for k, v in field["data"].items()}
        except PlatformError as exc:
            if exc.code == "expired":
                raise
        out: dict[int, dict] = {}
        page = 1
        while page <= max_pages and len(out) < len(product_ids):
            async with ctx.step(f"review_{page}", f"خواندن وضعیت محصولات، صفحهٔ {page}") as st:
                data = (await self._request("GET", f"/api/v2/products/seller?page={page}&size=50&order=desc&categoryId=",
                                            token=token, referer=f"{BASE_URL}/pwa/product-management")).get("data") or {}
                st.response = {"count": len(data.get("items") or [])}
            for row in data.get("items") or []:
                pid = row.get("product_id")
                if pid not in product_ids:
                    continue
                mod = row.get("moderation_status") if isinstance(row.get("moderation_status"), dict) else {}
                code = row.get("status_data") or mod.get("status") or mod.get("title") or ""
                code = code if isinstance(code, str) else ""
                state = review_state(code)
                if state == "approved" and int(row.get("variants_count") or 0) > 0:
                    state = "live"
                out[pid] = {"code": code, "label": labels.get(code) or REVIEW_FA.get(state, code), "state": state,
                            "variants_count": int(row.get("variants_count") or 0)}
            pager = data.get("pager") or {}
            if not data.get("items") or page >= int(pager.get("total_pages") or 1):
                break
            page += 1
        return out

    async def create_variant(self, ctx, channel: Channel, product_id: int, product: Product) -> dict[str, Any]:
        """Price + stock for an approved product: the panel's variant (تنوع) wizard. Price in rial, Digikala's unit."""
        token = self._token(channel)
        ref = f"{BASE_URL}/pwa/variant/create?productId={product_id}"
        if not product.price:
            raise PlatformError("قیمت محصول ثبت نشده؛ برای فروش در دیجی‌کالا قیمت لازم است.", "no_price", EDIT_PRODUCT)
        async with ctx.step("variant_form", "دریافت فرم تنوع دیجی‌کالا", request={"product_id": product_id}) as s:
            form = (await self._request("GET", f"/api/v2/variant-creation/v2/{product_id}", token=token, referer=ref)).get("data") or {}
            s.response = {k: (v if not isinstance(v, (list, dict)) else f"{len(v)} items") for k, v in form.items()} if isinstance(form, dict) else {}
        body = variant_body(form, product, product_id)
        async with ctx.step("variant", "ثبت قیمت و موجودی در دیجی‌کالا", request=body) as s:
            res = (await self._request("POST", "/api/v2/variant-creation/variants/", token=token, body=body, referer=ref)).get("data") or {}
            s.response = res
        vid = _first(res if isinstance(res, dict) else {}, "variant_id", "id", "variant.id", "variants.0.id")
        return {"variant_id": vid, "price_toman": product.price, "stock": product.stock_qty}

    async def fetch_orders(self, ctx, channel: Channel, pages: int = 3) -> list[dict[str, Any]]:
        """Newest orders of the seller (Digikala-warehouse and ship-by-seller), normalised; endpoints that this account
        can't use are skipped."""
        token = self._token(channel)
        out: dict[str, dict] = {}
        errors = []
        for path in ORDER_PATHS:
            for page in range(1, pages + 1):
                sep = "&" if "?" in path else "?"
                try:
                    async with ctx.step(f"orders{path.replace('/', '_')}_{page}", f"دریافت سفارش‌ها، صفحهٔ {page}",
                                        request={"path": path, "page": page}) as st:
                        payload = await self._request("GET", f"{path}{sep}page={page}&size=50", token=token,
                                                      referer=f"{BASE_URL}/pwa/orders")
                        data = payload.get("data")
                        st.response = {"keys": list(data)[:20] if isinstance(data, dict) else type(data).__name__}
                except PlatformError as exc:
                    if exc.code == "expired":
                        raise
                    errors.append(f"{path}: {exc.message_fa}")
                    break
                rows = _rows(data)
                for row in rows:
                    o = normalize_order(row)
                    if o:
                        o["kind"] = "ship_by_seller" if "ship-by-seller" in path else "digikala"
                        out.setdefault(o["id"], o)
                pager = (data or {}).get("pager") if isinstance(data, dict) else None
                if not rows or page >= int((pager or {}).get("total_pages") or 1):
                    break
        if not out and len(errors) == len(ORDER_PATHS):
            raise PlatformError("سفارش‌های دیجی‌کالا خوانده نشد.", "orders_failed", detail=errors)
        return list(out.values())

    async def fetch_items(self, ctx, channel: Channel, limit: int = 200) -> list[CrawledItem]:
        await self.ensure_enabled(ctx.db)
        token = self._token(channel)
        items: list[CrawledItem] = []
        page = 1
        while len(items) < limit:
            async with ctx.step(f"page_{page}", f"دریافت صفحهٔ {page} محصولات فروشنده",
                                request={"page": page, "size": 50}) as s:
                data = (await self._request("GET", f"/api/v2/products/seller?page={page}&size=50&order=desc&categoryId=",
                                            token=token, referer=f"{BASE_URL}/pwa/product-management")).get("data") or {}
                rows = data.get("items") or []
                s.response = {"count": len(rows), "pager": data.get("pager")}
            for row in rows:
                pid = row.get("product_id")
                cat = row.get("main_category_title")
                items.append(CrawledItem(
                    external_id=f"DKP-{pid}", title=row.get("title_fa") or row.get("title") or "",
                    brand=row.get("brand_title_fa") or "",
                    category=next(iter(cat.values()), "") if isinstance(cat, dict) and cat else "",
                    image_urls=[row["image_src"]] if row.get("image_src") else [],
                    url=row.get("product_url") or "", sku=f"DKP-{pid}", raw=row))
            pager = data.get("pager") or {}
            if not rows or page >= int(pager.get("total_pages") or 1):
                break
            page += 1
        return items[:limit]


digikala = DigikalaAdapter()
