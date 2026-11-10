"""The store's own website («سایت من»): settings for the seller, and the public API the storefront app reads.

- Seller side (`/stores/{id}/site`): enable the site, pick a subdomain of ours or connect an own domain, palette,
  banners, about, contact, footer badges (Enamad…), chat widget, SEO and shipping.
- Public side (`/public/sites/...`, no account): resolve a host to a site, catalog, categories, product pages, cart
  quote, checkout (an Invoice with source "site": it shows in the unified orders list and is paid on the existing
  public invoice page), sitemap.xml, and the chat widget, which runs on the same assistant as Telegram and Bale.
"""

import asyncio
import html
import re
import secrets
import socket
import time
from collections import defaultdict, deque
from datetime import timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.db import utcnow
from app.deps import DB, CurrentUser, get_store_for
from app.models import ChatConversation, ChatMessage, Invoice, Product, Store, StoreSite
from app.routers.orders import _latin_digits, _totals, invoice_settings, next_number
from app.services import chat_runtime as rt
from app.services.categories import store_categories
from app.services.chat_engine import _COLOR_KEY, _SIZE_KEY, get_agent, variant_options
from app.services.notify import notify

router = APIRouter(tags=["site"])

RESERVED = {"www", "api", "app", "admin", "mail", "static", "cdn", "media", "assets", "sello", "panel", "dashboard",
            "shop", "store", "help", "support", "blog", "status", "ns1", "ns2", "smtp", "ftp", "test", "dev"}
SUB_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,30}[a-z0-9])$")
DOMAIN_RE = re.compile(r"^(?=.{4,190}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
PHONE_RE = re.compile(r"^09\d{9}$")

DEFAULT_CONFIG: dict[str, Any] = {
    "title": "",  # shown in the header and as the home page title; empty = the store's name
    "tagline": "",
    "logo_url": "",  # empty = the store's logo
    "palette": {"preset": "emerald", "primary": "#28594d", "accent": "#c6a66b"},
    "banners": [],  # [{"image_url", "mobile_image_url", "title", "subtitle", "link", "button"}]
    "about": {"title": "درباره ما", "text": "", "image_url": ""},
    "contact": {"phone": "", "email": "", "address": "", "hours": "", "instagram": "", "telegram": "",
                "whatsapp": "", "map_url": ""},
    "badges": [],  # [{"kind": "enamad" | "samandehi" | "custom", "title", "href", "img"}]
    "chat": {"enabled": True, "greeting": "سلام! سؤالی دربارهٔ محصولات داری؟ همین‌جا بپرس.", "position": "left"},
    "seo": {"title": "", "description": "", "keywords": ""},
    "shipping": {"fee_toman": 0, "free_over_toman": 0, "note": ""},
    "home": {"show_categories": True, "show_new": True, "show_sale": True},
}


# ---------- helpers ----------


def _aware(d):
    return d if d is None or d.tzinfo else d.replace(tzinfo=timezone.utc)


def merged_config(site: StoreSite | None) -> dict[str, Any]:
    cfg = (site.config if site else None) or {}
    out: dict[str, Any] = {}
    for k, v in DEFAULT_CONFIG.items():
        got = cfg.get(k)
        out[k] = {**v, **(got or {})} if isinstance(v, dict) else (got if got is not None else v)
    return out


def site_url(site: StoreSite) -> str:
    """The site's canonical address: the seller's own domain once it points to us, else our subdomain."""
    scheme = settings.site_scheme
    if site.custom_domain and site.domain_status == "active":
        return f"{scheme}://{site.custom_domain}"
    if site.subdomain and settings.site_root_domain:
        return f"{scheme}://{site.subdomain}.{settings.site_root_domain}"
    if site.custom_domain:
        return f"{scheme}://{site.custom_domain}"
    return ""


def abs_media(url: str) -> str:
    """A stable absolute address for a /media URL (the backend route redirects to the bucket when there is one)."""
    if url and url.startswith("/media/"):
        return f"{settings.public_base_url.rstrip('/')}{url}"
    return url or ""


def slugify(title: str) -> str:
    s = re.sub(r"[^\w\s‌-]", " ", title or "", flags=re.UNICODE)
    s = re.sub(r"[\s‌_]+", "-", s.strip()).strip("-").lower()
    return s[:80].strip("-") or "p"


def product_path(p: Product) -> str:
    return f"/product/{p.id}/{quote(slugify(p.title))}"


def category_path(name: str) -> str:
    return f"/category/{quote(name)}"


def _specs(p: Product, opts: dict[str, list[str]]) -> list[dict[str, str]]:
    """Product specs for the spec table: attributes that are not a choice the customer makes (size, colour...)."""
    out = []
    for k, v in (p.attributes or {}).items():
        key = str(k).strip()
        if not str(v).strip() or len(out) >= 20:
            continue
        if opts and (_SIZE_KEY.search(key) or _COLOR_KEY.search(key) or key[:30] in opts):
            continue
        out.append({"name": key[:60], "value": str(v)[:300]})
    return out


def product_card(p: Product) -> dict[str, Any]:
    imgs = [abs_media(i.url) for i in sorted(p.images or [], key=lambda i: i.position)]
    return {"id": p.id, "title": p.title, "price": int(p.price or 0),
            "compare_at_price": int(p.compare_at_price or 0) if (p.compare_at_price or 0) > (p.price or 0) else 0,
            "in_stock": (p.stock_qty or 0) > 0, "few_left": 0 < (p.stock_qty or 0) <= 3,
            "image": imgs[0] if imgs else "", "image2": imgs[1] if len(imgs) > 1 else "",
            "category": p.category or "", "path": product_path(p)}


def product_full(p: Product) -> dict[str, Any]:
    opts = variant_options(p)
    labels = {"size": "سایز", "color": "رنگ"}
    return {**product_card(p), "description": p.description or "", "brand": p.brand or "", "sku": p.sku or "",
            "images": [abs_media(i.url) for i in sorted(p.images or [], key=lambda i: i.position)],
            "options": [{"key": k, "label": labels.get(k, k), "values": v} for k, v in opts.items()],
            "specs": _specs(p, opts),
            "tags": p.tags or [], "stock_qty": max(0, p.stock_qty or 0) if 0 < (p.stock_qty or 0) <= 3 else None,
            "updated_at": p.updated_at, "category_path": category_path(p.category) if p.category else ""}


def _public_products(store_id: int):
    return select(Product).where(Product.store_id == store_id, Product.status == "ready")


# Simple per-address limits for the public write endpoints (one process; enough against casual abuse).
_hits: dict[str, deque] = defaultdict(deque)


def _limit(request: Request, bucket: str, n: int, seconds: int) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0]
    q = _hits[f"{bucket}:{ip.strip()}"]
    now = time.monotonic()
    while q and q[0] < now - seconds:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, "درخواست‌ها زیاد شد؛ کمی بعد دوباره امتحان کن.")
    q.append(now)


def parse_badge(kind: str, title: str, code: str) -> dict[str, str]:
    """A trust badge from the code the issuer gives (Enamad, Samandehi...) or a plain link: only an https link and an
    https image are kept, so no script from the pasted code runs on the site."""
    code = html.unescape(code or "").strip()
    href = re.search(r"""href\s*=\s*["']([^"']+)["']""", code, re.I)
    img = re.search(r"""src\s*=\s*["']([^"']+)["']""", code, re.I)
    href_v = href.group(1) if href else (code if code.startswith("https://") and " " not in code else "")
    img_v = img.group(1) if img else ""
    if not href_v.startswith("https://") or (img_v and not img_v.startswith("https://")):
        raise HTTPException(422, "کد نماد باید لینک و عکس https داشته باشد؛ کد را کامل از سایت صادرکننده کپی کن.")
    if kind == "enamad" and not img_v:
        m = re.search(r"[?&]id=(\d+)&(?:amp;)?Code=([\w]+)", href_v, re.I)
        if m:
            img_v = f"https://trustseal.enamad.ir/logo.aspx?id={m.group(1)}&Code={m.group(2)}"
    return {"kind": kind, "title": title or {"enamad": "اینماد", "samandehi": "ساماندهی"}.get(kind, "نماد"),
            "href": href_v[:600], "img": img_v[:600]}


# ---------- seller side ----------


class BannerIn(BaseModel):
    image_url: str = Field(max_length=1000)
    mobile_image_url: str = Field("", max_length=1000)
    title: str = Field("", max_length=120)
    subtitle: str = Field("", max_length=240)
    link: str = Field("", max_length=500)
    button: str = Field("", max_length=40)


class BadgeIn(BaseModel):
    kind: str = "custom"  # enamad | samandehi | custom
    title: str = Field("", max_length=60)
    code: str = Field("", max_length=4000)  # the pasted code or link
    href: str = Field("", max_length=600)  # already parsed (sent back as saved)
    img: str = Field("", max_length=600)


class SiteIn(BaseModel):
    enabled: bool | None = None
    subdomain: str | None = Field(None, max_length=40)
    custom_domain: str | None = Field(None, max_length=190)
    title: str | None = Field(None, max_length=80)
    tagline: str | None = Field(None, max_length=160)
    logo_url: str | None = Field(None, max_length=1000)
    palette: dict[str, str] | None = None
    banners: list[BannerIn] | None = Field(None, max_length=8)
    about: dict[str, str] | None = None
    contact: dict[str, str] | None = None
    badges: list[BadgeIn] | None = Field(None, max_length=6)
    chat: dict[str, Any] | None = None
    seo: dict[str, str] | None = None
    shipping: dict[str, Any] | None = None
    home: dict[str, bool] | None = None


async def _site(db, store: Store, create: bool = True) -> StoreSite | None:
    site = (await db.execute(select(StoreSite).where(StoreSite.store_id == store.id))).scalar_one_or_none()
    if not site and create:
        site = StoreSite(store_id=store.id, enabled=False, config={})
        db.add(site)
        await db.flush()
    return site


async def _seller_out(db, store: Store, site: StoreSite) -> dict[str, Any]:
    pay = await invoice_settings(db, store)
    agent = await get_agent(db, store.id)
    count = await db.scalar(select(func.count()).select_from(_public_products(store.id).subquery())) or 0
    return {"enabled": site.enabled, "subdomain": site.subdomain or "", "custom_domain": site.custom_domain or "",
            "domain_status": site.domain_status or "", "domain_checked_at": site.domain_checked_at,
            "url": site_url(site), "root_domain": settings.site_root_domain, "server_ip": settings.site_server_ip,
            "suggested_subdomain": f"shop{store.id}", "config": merged_config(site),
            "payment_ready": bool(pay["card_number"] or pay["sheba"] or pay["pay_link"]),
            "assistant_enabled": bool(agent.enabled), "products": int(count),
            "store": {"name": store.name, "logo_url": store.logo_url}}


def _clean_text(d: dict[str, Any] | None, keys: tuple[str, ...], limit: int = 500) -> dict[str, str]:
    return {k: str((d or {}).get(k) or "").strip()[:limit] for k in keys if k in (d or {})}


@router.get("/stores/{store_id}/site")
async def get_site(store_id: int, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    site = await _site(db, store)
    await db.commit()
    return await _seller_out(db, store, site)


@router.put("/stores/{store_id}/site")
async def put_site(store_id: int, body: SiteIn, user: CurrentUser, db: DB):
    store = await get_store_for(db, user, store_id)
    site = await _site(db, store)
    cfg = dict(site.config or {})

    if body.subdomain is not None:
        sub = body.subdomain.strip().lower()
        if sub:
            if not SUB_RE.match(sub) or "--" in sub:
                raise HTTPException(422, "زیردامنه فقط حروف انگلیسی کوچک، عدد و خط تیره (۳ تا ۳۲ حرف) باشد.")
            if sub in RESERVED:
                raise HTTPException(422, "این زیردامنه رزرو است؛ اسم دیگری انتخاب کن.")
            taken = await db.scalar(select(StoreSite.id).where(StoreSite.subdomain == sub, StoreSite.id != site.id))
            if taken:
                raise HTTPException(409, "این زیردامنه قبلاً گرفته شده.")
        site.subdomain = sub or None
    if body.custom_domain is not None:
        dom = body.custom_domain.strip().lower()
        dom = re.sub(r"^https?://", "", dom).split("/")[0].removeprefix("www.")
        if dom:
            if not DOMAIN_RE.match(dom):
                raise HTTPException(422, "دامنه معتبر نیست؛ مثل myshop.ir بنویس.")
            if settings.site_root_domain and (dom == settings.site_root_domain or dom.endswith("." + settings.site_root_domain)):
                raise HTTPException(422, "برای آدرس روی دامنهٔ ما از بخش زیردامنه استفاده کن.")
            taken = await db.scalar(select(StoreSite.id).where(StoreSite.custom_domain == dom, StoreSite.id != site.id))
            if taken:
                raise HTTPException(409, "این دامنه به فروشگاه دیگری وصل است.")
        if dom != (site.custom_domain or ""):
            site.domain_status = "pending" if dom else ""
            site.domain_checked_at = None
        site.custom_domain = dom or None

    for k in ("title", "tagline", "logo_url"):
        if (v := getattr(body, k)) is not None:
            cfg[k] = v.strip()
    if body.palette is not None:
        pal = {**DEFAULT_CONFIG["palette"], **(cfg.get("palette") or {})}
        for k in ("primary", "accent"):
            if k in body.palette:
                if not HEX_RE.match(body.palette[k] or ""):
                    raise HTTPException(422, "رنگ باید کد هگز مثل #28594d باشد.")
                pal[k] = body.palette[k].lower()
        if "preset" in body.palette:
            pal["preset"] = str(body.palette["preset"])[:20]
        cfg["palette"] = pal
    if body.banners is not None:
        cfg["banners"] = [b.model_dump() for b in body.banners if b.image_url.strip()]
    if body.about is not None:
        cfg["about"] = {**(cfg.get("about") or {}), **_clean_text(body.about, ("title",), 80),
                        **_clean_text(body.about, ("text",), 8000), **_clean_text(body.about, ("image_url",), 1000)}
    if body.contact is not None:
        c = _clean_text(body.contact, tuple(DEFAULT_CONFIG["contact"].keys()), 500)
        if "phone" in c:
            c["phone"] = _latin_digits(c["phone"])
        cfg["contact"] = {**(cfg.get("contact") or {}), **c}
    if body.badges is not None:
        badges = []
        for b in body.badges:
            if b.code.strip():
                badges.append(parse_badge(b.kind, b.title.strip(), b.code))
            elif b.href.startswith("https://") and (not b.img or b.img.startswith("https://")):
                badges.append({"kind": b.kind, "title": b.title.strip() or "نماد", "href": b.href, "img": b.img})
        cfg["badges"] = badges
    if body.chat is not None:
        ch = {**DEFAULT_CONFIG["chat"], **(cfg.get("chat") or {})}
        if "enabled" in body.chat:
            ch["enabled"] = bool(body.chat["enabled"])
        if "greeting" in body.chat:
            ch["greeting"] = str(body.chat["greeting"] or "").strip()[:300]
        if body.chat.get("position") in ("left", "right"):
            ch["position"] = body.chat["position"]
        cfg["chat"] = ch
    if body.seo is not None:
        cfg["seo"] = {**(cfg.get("seo") or {}), **_clean_text(body.seo, ("title",), 70),
                      **_clean_text(body.seo, ("description",), 320), **_clean_text(body.seo, ("keywords",), 300)}
    if body.shipping is not None:
        sh = {**DEFAULT_CONFIG["shipping"], **(cfg.get("shipping") or {})}
        for k in ("fee_toman", "free_over_toman"):
            if k in body.shipping:
                sh[k] = max(0, int(body.shipping[k] or 0))
        if "note" in body.shipping:
            sh["note"] = str(body.shipping["note"] or "").strip()[:300]
        cfg["shipping"] = sh
    if body.home is not None:
        cfg["home"] = {**DEFAULT_CONFIG["home"], **(cfg.get("home") or {}),
                       **{k: bool(v) for k, v in body.home.items() if k in DEFAULT_CONFIG["home"]}}
    site.config = cfg

    if body.enabled is not None:
        if body.enabled and not site.subdomain and not site.custom_domain:
            site.subdomain = f"shop{store.id}"
        site.enabled = body.enabled
    await db.commit()
    return await _seller_out(db, store, site)


def _resolve_ips(host: str) -> set[str]:
    try:
        return {a[4][0] for a in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)}
    except OSError:
        return set()


@router.post("/stores/{store_id}/site/check-domain")
async def check_domain(store_id: int, user: CurrentUser, db: DB):
    """Whether the seller's own domain points to our server yet (its A record)."""
    store = await get_store_for(db, user, store_id)
    site = await _site(db, store)
    if not site.custom_domain:
        raise HTTPException(422, "اول دامنه را وارد کن.")
    ips = await asyncio.to_thread(_resolve_ips, site.custom_domain)
    want = settings.site_server_ip.strip()
    ok = bool(ips) and (not want or want in ips)
    site.domain_status = "active" if ok else "pending"
    site.domain_checked_at = utcnow()
    await db.commit()
    out = await _seller_out(db, store, site)
    out["dns"] = sorted(ips)
    out["message"] = ("دامنه درست به سرور ما وصل است." if ok else
                      "دامنه هنوز به سرور ما اشاره نمی‌کند؛ رکورد A را بساز و بعد از چند دقیقه دوباره بررسی کن."
                      if ips else "این دامنه هنوز رکورد DNS ندارد.")
    return out


# ---------- public: resolving a host ----------


async def site_for_host(db, host: str) -> StoreSite | None:
    host = (host or "").lower().split(":")[0].strip().removeprefix("www.")
    if not host:
        return None
    root = settings.site_root_domain.lower()
    sub = ""
    if root and host.endswith("." + root):
        sub = host[: -len(root) - 1]
    elif host.endswith(".localhost"):
        sub = host.removesuffix(".localhost")
    stmt = select(StoreSite).where(StoreSite.subdomain == sub) if sub else select(StoreSite).where(
        StoreSite.custom_domain == host)
    return (await db.execute(stmt)).scalar_one_or_none()


async def _public_site(db, key: str) -> tuple[StoreSite, Store]:
    site = (await db.execute(select(StoreSite).where(StoreSite.subdomain == key.lower()))).scalar_one_or_none()
    store = await db.get(Store, site.store_id) if site else None
    if not site or not site.enabled or not store or store.archived:
        raise HTTPException(404, "این سایت پیدا نشد.")
    return site, store


def _site_key(site: StoreSite) -> str:
    return site.subdomain or f"shop{site.store_id}"


async def _site_payload(db, site: StoreSite, store: Store) -> dict[str, Any]:
    cfg = merged_config(site)
    cats = await store_categories(db, store.id)
    # Only categories with a product on sale count (store_categories also counts drafts).
    live = dict((await db.execute(select(Product.category, func.count()).where(
        Product.store_id == store.id, Product.status == "ready", Product.category.is_not(None), Product.category != "")
        .group_by(Product.category))).all())
    agent = await get_agent(db, store.id)
    pay = await invoice_settings(db, store)
    cfg["banners"] = [{**b, "image_url": abs_media(b.get("image_url", "")),
                       "mobile_image_url": abs_media(b.get("mobile_image_url", ""))} for b in cfg["banners"]]
    cfg["about"]["image_url"] = abs_media(cfg["about"].get("image_url", ""))
    cfg["chat"]["enabled"] = bool(cfg["chat"].get("enabled")) and bool(agent.enabled)
    return {"key": _site_key(site), "store_id": store.id, "name": cfg["title"] or store.name, "store_name": store.name,
            "logo_url": abs_media(cfg["logo_url"] or store.logo_url or ""), "url": site_url(site),
            "categories": [{"name": c["name"], "count": live[c["name"]], "path": category_path(c["name"])}
                           for c in cats if live.get(c["name"])],
            "checkout_ready": bool(pay["card_number"] or pay["sheba"] or pay["pay_link"]),
            "ask_address": bool(pay.get("ask_address", True)), "terms": pay.get("note") or "",
            "config": cfg}


@router.get("/public/sites/resolve")
async def resolve(db: DB, host: str = Query(..., max_length=255)):
    """The storefront calls this with the request's Host header to know which store's site to render."""
    site = await site_for_host(db, host)
    store = await db.get(Store, site.store_id) if site else None
    if not site or not site.enabled or not store or store.archived:
        raise HTTPException(404, "این سایت پیدا نشد.")
    return await _site_payload(db, site, store)


@router.get("/public/sites/tls-check")
async def tls_check(db: DB, domain: str = Query("", max_length=255)):
    """For the web server's on-demand TLS: a certificate is issued only for hosts that are a live store site."""
    site = await site_for_host(db, domain)
    if not site or not site.enabled:
        raise HTTPException(404, "unknown")
    return {"ok": True}


@router.get("/public/sites/{key}")
async def public_site(key: str, db: DB):
    site, store = await _public_site(db, key)
    return await _site_payload(db, site, store)


# ---------- public: catalog ----------


@router.get("/public/sites/{key}/products")
async def public_products(key: str, db: DB, q: str = Query("", max_length=100), category: str = Query("", max_length=120),
                          sort: str = "new", available: bool = False, sale: bool = False,
                          page: int = Query(1, ge=1, le=1000), size: int = Query(24, ge=1, le=48)):
    site, store = await _public_site(db, key)
    stmt = _public_products(store.id)
    if q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(Product.title.ilike(like), Product.description.ilike(like), Product.brand.ilike(like),
                              Product.category.ilike(like)))
    if category:
        stmt = stmt.where(Product.category == category)
    if available:
        stmt = stmt.where(Product.stock_qty > 0)
    if sale:
        stmt = stmt.where(Product.compare_at_price > Product.price)
    total = int(await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
    order = {"price": [Product.price.asc()], "-price": [Product.price.desc()]}.get(sort, [Product.created_at.desc()])
    # In-stock products first, then the chosen order.
    in_stock = (Product.stock_qty > 0).desc()
    rows = (await db.execute(stmt.options(selectinload(Product.images)).order_by(in_stock, *order, Product.id.desc())
                             .offset((page - 1) * size).limit(size))).scalars().all()
    return {"total": total, "page": page, "size": size, "pages": max(1, -(-total // size)),
            "items": [product_card(p) for p in rows]}


@router.get("/public/sites/{key}/products/{product_id}")
async def public_product(key: str, product_id: int, db: DB):
    site, store = await _public_site(db, key)
    p = (await db.execute(_public_products(store.id).where(Product.id == product_id)
                          .options(selectinload(Product.images)))).scalar_one_or_none()
    if not p:
        raise HTTPException(404, "محصول پیدا نشد.")
    rel = select(Product).where(Product.store_id == store.id, Product.status == "ready", Product.id != p.id,
                                Product.stock_qty > 0)
    rel = rel.where(Product.category == p.category) if p.category else rel
    related = (await db.execute(rel.options(selectinload(Product.images)).order_by(Product.created_at.desc())
                                .limit(8))).scalars().all()
    return {**product_full(p), "related": [product_card(r) for r in related]}


# ---------- public: cart and checkout ----------


class CartLine(BaseModel):
    product_id: int
    qty: int = Field(1, ge=1, le=50)
    variant: str = Field("", max_length=120)


class QuoteIn(BaseModel):
    items: list[CartLine] = Field(min_length=1, max_length=30)


class CheckoutIn(QuoteIn):
    name: str = Field(min_length=2, max_length=160)
    phone: str = Field(max_length=30)
    address: str = Field("", max_length=1000)
    postal_code: str = Field("", max_length=20)
    note: str = Field("", max_length=1000)


async def _quote(db, store: Store, site: StoreSite, items: list[CartLine]) -> dict[str, Any]:
    ids = {i.product_id for i in items}
    products = {p.id: p for p in (await db.execute(_public_products(store.id).where(Product.id.in_(ids))
                                                   .options(selectinload(Product.images)))).scalars()}
    lines, problems = [], []
    for it in items:
        p = products.get(it.product_id)
        if not p:
            problems.append({"product_id": it.product_id, "message": "این محصول دیگر فروخته نمی‌شود."})
            continue
        stock = max(0, p.stock_qty or 0)
        qty = min(it.qty, stock)
        if stock <= 0:
            problems.append({"product_id": p.id, "message": f"«{p.title}» ناموجود شد."})
        elif qty < it.qty:
            problems.append({"product_id": p.id, "message": f"از «{p.title}» فقط {stock} عدد موجود است."})
        opts = variant_options(p)
        lines.append({"product_id": p.id, "title": p.title, "variant": it.variant.strip(), "qty": it.qty,
                      "available_qty": qty, "unit_price": int(p.price or 0), "image_url": product_card(p)["image"],
                      "path": product_path(p), "needs_variant": bool(opts) and not it.variant.strip()})
    sh = merged_config(site)["shipping"]
    subtotal = sum(ln["unit_price"] * ln["available_qty"] for ln in lines)
    free = sh.get("free_over_toman") or 0
    shipping = 0 if not subtotal or (free and subtotal >= free) else int(sh.get("fee_toman") or 0)
    return {"items": lines, "problems": problems, "subtotal_toman": subtotal, "shipping_toman": shipping,
            "total_toman": subtotal + shipping, "free_shipping_over": free, "shipping_note": sh.get("note") or ""}


@router.post("/public/sites/{key}/cart/quote")
async def cart_quote(key: str, body: QuoteIn, db: DB):
    """Today's prices and stock for the cart kept in the browser."""
    site, store = await _public_site(db, key)
    return await _quote(db, store, site, body.items)


@router.post("/public/sites/{key}/checkout", status_code=201)
async def checkout(key: str, body: CheckoutIn, request: Request, db: DB):
    """Places the order: an Invoice (source "site") that shows in the seller's orders list at once. The customer then
    pays card to card (or the seller's payment link) and sends the receipt on the order page (/order/<code>)."""
    _limit(request, "checkout", 8, 600)
    site, store = await _public_site(db, key)
    conf = await invoice_settings(db, store)
    if not (conf["card_number"] or conf["sheba"] or conf["pay_link"]):
        raise HTTPException(409, "فروشگاه هنوز روش پرداخت را تنظیم نکرده؛ لطفاً از راه تماس با فروشگاه سفارش بده.")
    phone = _latin_digits(body.phone).replace(" ", "").replace("+98", "0")
    if phone.startswith("98") and len(phone) == 12:
        phone = "0" + phone[2:]
    if not PHONE_RE.match(phone):
        raise HTTPException(422, {"message": "شماره موبایل درست نیست؛ مثل ۰۹۱۲۱۲۳۴۵۶۷ بنویس.", "fields": ["phone"]})
    if conf.get("ask_address", True) and len(body.address.strip()) < 8:
        raise HTTPException(422, {"message": "آدرس کامل تحویل را بنویس.", "fields": ["address"]})
    q = await _quote(db, store, site, body.items)
    if q["problems"]:
        raise HTTPException(409, {"message": q["problems"][0]["message"], "quote": q})
    if any(ln["needs_variant"] for ln in q["items"]):
        raise HTTPException(422, "برای بعضی محصولات سایز یا رنگ انتخاب نشده.")
    address = body.address.strip() + (f"\nکد پستی: {_latin_digits(body.postal_code)}" if body.postal_code.strip() else "")
    hours = int(conf.get("expire_hours") or 0)
    inv = Invoice(
        store_id=store.id, number=await next_number(db, store, conf), source="site", code=secrets.token_urlsafe(9),
        platform="site", customer_name=body.name.strip(), customer_phone=phone, customer_address=address,
        items=[{k: ln[k] for k in ("product_id", "title", "variant", "qty", "unit_price", "image_url")}
               for ln in q["items"]],
        # The customer's own note goes where the seller reads it on the order card.
        discount_toman=0, shipping_toman=q["shipping_toman"], note=body.note.strip(), status="awaiting_payment",
        submitted_at=utcnow(),
        expires_at=utcnow() + timedelta(hours=hours) if hours else None,
    )
    _totals(inv)
    db.add(inv)
    await db.flush()
    await notify(db, store.id, "approval", f"سفارش تازه از سایت · {inv.number}",
                 f"{inv.customer_name} · {inv.total_toman:,} تومان. منتظر پرداخت مشتری.",
                 link=f"/app/s/{store.id}/orders?o=inv-{inv.id}", image_url=inv.items[0].get("image_url") or "")
    await db.commit()
    return {"code": inv.code, "number": inv.number, "total_toman": inv.total_toman}


# ---------- public: sitemap ----------


def _x(s: str) -> str:
    return html.escape(s, quote=True)


@router.get("/public/sites/{key}/sitemap.xml", include_in_schema=False)
async def sitemap(key: str, db: DB, base: str = Query("", max_length=255)):
    """Every page of the site for search engines: home, catalog, categories, about, contact and each product with its
    pictures. `base` is the address the storefront is being visited on (defaults to the site's canonical one), so the
    links in it always match the host search engines fetched it from."""
    site, store = await _public_site(db, key)
    root = base or site_url(site)
    p = urlparse(root)
    root = f"{p.scheme or 'https'}://{p.netloc or p.path}".rstrip("/")
    rows = (await db.execute(_public_products(store.id).options(selectinload(Product.images))
                             .order_by(Product.updated_at.desc()).limit(45000))).scalars().all()
    newest = max((r.updated_at for r in rows), default=site.updated_at)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
           'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">']

    def url(path: str, lastmod=None, freq: str = "weekly", prio: str = "0.5", images: list[str] | None = None,
            title: str = "") -> None:
        parts = [f"<url><loc>{_x(root + path)}</loc>"]
        if lastmod:
            parts.append(f"<lastmod>{_aware(lastmod).date().isoformat()}</lastmod>")
        parts.append(f"<changefreq>{freq}</changefreq><priority>{prio}</priority>")
        for img in (images or [])[:5]:
            parts.append(f"<image:image><image:loc>{_x(img)}</image:loc>"
                         + (f"<image:title>{_x(title)}</image:title>" if title else "") + "</image:image>")
        parts.append("</url>")
        out.append("".join(parts))

    url("/", newest, "daily", "1.0")
    url("/products", newest, "daily", "0.8")
    for c in (await _site_payload(db, site, store))["categories"]:
        url(c["path"], newest, "daily", "0.7")
    for r in rows:
        url(product_path(r), r.updated_at, "weekly", "0.9",
            [abs_media(i.url) for i in sorted(r.images or [], key=lambda i: i.position)], r.title)
    url("/about", site.updated_at, "monthly", "0.3")
    url("/contact", site.updated_at, "monthly", "0.3")
    out.append("</urlset>")
    return Response("\n".join(out), media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=600"})


# ---------- public: chat widget ----------


class WidgetIn(BaseModel):
    visitor: str = Field("", max_length=40)
    text: str = Field(min_length=1, max_length=1000)
    name: str = Field("", max_length=80)
    page: str = Field("", max_length=300)  # the page the visitor is on, for the seller's context


def _widget_msg(m: ChatMessage) -> dict[str, Any]:
    return {"id": m.id, "role": m.role, "text": m.text, "image_url": abs_media(m.image_url),
            "buttons": m.buttons or [], "created_at": m.created_at}


async def _widget_conv(db, store: Store, visitor: str) -> ChatConversation | None:
    if not visitor:
        return None
    return (await db.execute(select(ChatConversation).where(
        ChatConversation.store_id == store.id, ChatConversation.platform == "site",
        ChatConversation.external_chat_id == visitor))).scalar_one_or_none()


@router.get("/public/sites/{key}/chat")
async def widget_history(key: str, db: DB, visitor: str = Query("", max_length=40), after: int = 0):
    """The visitor's chat (polled while the widget is open, for the seller's replies)."""
    site, store = await _public_site(db, key)
    conv = await _widget_conv(db, store, visitor)
    if not conv:
        return {"visitor": visitor, "messages": []}
    rows = (await db.execute(select(ChatMessage).where(
        ChatMessage.conversation_id == conv.id, ChatMessage.id > after,
        ChatMessage.role.in_(["customer", "ai", "seller"])).order_by(ChatMessage.id).limit(100))).scalars().all()
    return {"visitor": visitor, "messages": [_widget_msg(m) for m in rows]}


@router.post("/public/sites/{key}/chat")
async def widget_send(key: str, body: WidgetIn, request: Request, db: DB):
    """A message from the site's chat widget: saved in the store's chats like a Telegram message and answered by the
    same assistant (or by the seller from the panel)."""
    _limit(request, "chat", 40, 300)
    site, store = await _public_site(db, key)
    if not merged_config(site)["chat"].get("enabled"):
        raise HTTPException(403, "چت این سایت خاموش است.")
    visitor = body.visitor if re.fullmatch(r"[\w-]{16,40}", body.visitor or "") else secrets.token_urlsafe(16)
    conv = await _widget_conv(db, store, visitor)
    if not conv:
        conv = ChatConversation(store_id=store.id, bot_id=None, platform="site", mode="web", external_chat_id=visitor,
                                customer_name=body.name.strip() or "بازدیدکنندهٔ سایت", state={})
        db.add(conv)
        await db.flush()
    elif body.name.strip():
        conv.customer_name = body.name.strip()
    st = dict(conv.state or {})
    if body.page:
        st["site_page"] = body.page[:300]
        conv.state = st
    customer = rt.add_message(db, conv, role="customer", text=body.text.strip())
    conv.seller_unread = True
    await db.commit()
    replies = await rt.answer(conv.id, send=False)
    return {"visitor": visitor, "messages": [_widget_msg(customer)] + [_widget_msg(m) for m in replies]}
