"""Digikala seller panel. HTTP calls ported from platform-wrappers
(digikala.api/app/services/digikala_seller_client.py and digikala_seller_products.py).
These are the seller PWA's internal endpoints, not a public API, so they can change."""

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
from app.services.images import jpeg_bytes

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
            async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers=headers) as client:
                if files:
                    resp = await client.request(method, f"{BASE_URL}{path}", files=files, data=data)
                else:
                    resp = await client.request(method, f"{BASE_URL}{path}", json=body)
        except httpx.HTTPError as exc:
            raise PlatformError("ارتباط با پنل فروشندگان دیجی‌کالا برقرار نشد.", "network", detail=str(exc)) from exc
        if resp.status_code == 401:
            raise PlatformError("نشست دیجی‌کالا منقضی شده؛ دوباره با کد تأیید وارد شوید.", "expired", RECONNECT,
                                resp.text[:500])
        if resp.status_code >= 400:
            msg = self._extract_message(resp)
            raise PlatformError(f"دیجی‌کالا خطا داد: {msg}" if msg else "دیجی‌کالا درخواست را رد کرد.",
                                f"http_{resp.status_code}", detail=resp.text[:1000])
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

    async def publish(self, ctx, channel: Channel, product: Product, store: Store,
                      existing_remote_id: str = "") -> PublishResult:
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

        async with ctx.step("title", "ثبت عنوان و توضیح", request={"title_fa": title}) as s:
            auto = await self._request("GET", f"/api/v2/product-creation/{draft_id}/auto-title", token=token, referer=ref)
            body = {"draft_product_id": draft_id, "title_fa": title[:255], "title_en": "",
                    "description": desc if len(desc) >= 150 else None, "advantages": [], "disadvantages": []}
            s.response = {"auto_title": (auto.get("data") or {}).get("title_fa"),
                          "save": await self._request("POST", "/api/v2/product-creation/auto-title/save",
                                                      token=token, body=body, referer=ref)}

        async with ctx.step("images", f"آپلود {min(len(product.images), 8)} عکس") as s:
            ids = []
            for i, img in enumerate(product.images[:8]):
                data, name = await jpeg_bytes(img.url)
                up = await self._request("POST", "/api/v2/product-creation/temp_images", token=token, referer=ref,
                                         files={"file": (name, data, "image/jpeg")},
                                         data={"reference": "draft_product", "reference_id": str(draft_id)})
                image_id = (up.get("data") or {}).get("id") or ((up.get("data") or {}).get("image") or {}).get("id")
                if not image_id:
                    raise PlatformError("دیجی‌کالا عکس را نپذیرفت.", "image_rejected", EDIT_PRODUCT, up)
                ids.append(image_id)
                if i == 0:
                    await self._request("PATCH", f"/api/v2/product-creation/temp_images/{image_id}/main",
                                        token=token, body={}, referer=ref)
            s.response = {"image_ids": ids}

        async with ctx.step("save", "ثبت نهایی محصول") as s:
            saved = await self._request("POST", "/api/v2/product-creation/save", token=token, referer=ref,
                                        body={"category_id": cat_id, "draft_product_id": str(draft_id),
                                              "only_b2b": False, "use_temp_images": True})
            data = saved.get("data") or {}
            product_id = data.get("product_id") or (data.get("product") or {}).get("id") or data.get("id")
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

    async def _pick_category(self, ctx, token: str, product: Product) -> tuple[dict, dict]:
        keywords = [k for k in (product.category_hint, " ".join(product.title.split()[:2]), product.title.split()[0]) if k]
        candidates: list[dict] = []
        async with ctx.step("category_search", "جستجوی دستهٔ مناسب در دیجی‌کالا", request={"keywords": keywords}) as s:
            for kw in dict.fromkeys(keywords):
                q = httpx.QueryParams({"keyword": kw})
                data = await self._request("GET", f"/api/v2/product-creation/search/category/v2/{quote(kw)}?{q}",
                                           token=token, referer=f"{BASE_URL}/pwa/product/create")
                for it in (data.get("data") or {}).get("items") or []:
                    if it.get("id") and it.get("leaf", True) and all(c["id"] != it["id"] for c in candidates):
                        candidates.append({"id": it["id"], "title": it.get("text") or it.get("title") or "",
                                           "path": it.get("path") or ""})
                if len(candidates) >= 5:
                    break
            s.response = {"candidates": candidates[:10]}
        if not candidates:
            raise PlatformError("دستهٔ مناسبی در دیجی‌کالا برای این محصول پیدا نشد؛ «دسته» را در مشخصات محصول دقیق‌تر بنویسید.",
                                "no_category", EDIT_PRODUCT)

        ordered = candidates[:10]
        if len(ordered) > 1 and await ai.ai_available("field_fill"):
            async with ctx.step("category_ai", "انتخاب دسته با هوش مصنوعی") as s:
                res = await ai.generate_json(
                    task="field_fill", store_id=product.store_id,
                    system="You pick the best Digikala leaf category for a product. Reply JSON {\"id\": <id>}.",
                    prompt=json.dumps({"product": {"title": product.title, "category_hint": product.category_hint,
                                                   "description": (product.description or "")[:500]},
                                       "categories": ordered}, ensure_ascii=False))
                best = next((c for c in ordered if str(c["id"]) == str(res.get("id"))), None)
                if best:
                    ordered = [best] + [c for c in ordered if c is not best]
                s.response = {"picked": best}

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
        for f in fields:
            v = mine.get(_norm(f["title"]))
            if v not in (None, ""):
                values[str(f["id"])] = v
        missing = [f for f in fields if f["required"] and str(f["id"]) not in values]
        if missing and await ai.ai_available("field_fill"):
            async with ctx.step("attributes_ai", f"پر کردن {len(missing)} ویژگی اجباری با هوش مصنوعی") as s:
                res = await ai.generate_json(
                    task="field_fill", store_id=product.store_id,
                    system=("Fill Digikala product attributes from the product text. For select fields answer with one "
                            "option id, for multi_select a list of option ids, for number a number, else short Persian "
                            "text. Never invent specs that contradict the text; choose the most neutral option. "
                            "Reply JSON {\"<field id>\": value}."),
                    prompt=json.dumps({"product": {"title": product.title, "description": (product.description or "")[:1500],
                                                   "brand": product.brand, "attributes": product.attributes},
                                       "fields": [{k: f[k] for k in ("id", "title", "type", "unit", "options")} for f in missing]},
                                      ensure_ascii=False))
                values.update({str(k): v for k, v in res.items()})
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

    async def fetch_items(self, ctx, channel: Channel, limit: int = 200) -> list[CrawledItem]:
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
