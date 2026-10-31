"""Public Instagram profile crawl by username, through a scraping provider (Instagram has no API for other
people's posts). The provider is set by the admin; only Apify is wired today, behind `fetch_posts`, so another
provider (e.g. HikerAPI) can be added without touching the crawl job.

Rules the seller asked for: only the newest 25 posts; image posts (single photo or carousel) are taken, reels are
skipped unless the source allows them, and then only the reel's cover image is kept."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from app.platforms.base import PlatformError

LIMIT = 25
APIFY_ACTOR = "apify~instagram-scraper"
_USERNAME = re.compile(r"^[A-Za-z0-9._]{1,30}$")


@dataclass
class IgPost:
    shortcode: str
    kind: str  # image | carousel | reel
    caption: str = ""
    url: str = ""
    image_urls: list[str] = field(default_factory=list)  # for a reel: just its cover
    taken_at: str = ""


def normalize_username(raw: str) -> str:
    """'@shop', 'shop', 'https://instagram.com/shop/?hl=fa' -> 'shop'. Empty when it isn't a username."""
    s = (raw or "").strip()
    m = re.search(r"instagram\.com/([^/?#]+)", s)
    s = (m.group(1) if m else s).lstrip("@").strip("/").lower()
    return s if _USERNAME.match(s) and s not in ("p", "reel", "reels", "explore", "stories") else ""


async def fetch_posts(cfg: dict, username: str) -> list[IgPost]:
    if not cfg.get("token"):
        raise PlatformError("کرال اینستاگرام هنوز راه‌اندازی نشده؛ ادمین باید کلید سرویس کرال را در پنل ادمین وارد کند.",
                            "ig_crawler_not_configured")
    if cfg.get("provider", "apify") != "apify":
        raise PlatformError("سرویس کرال اینستاگرام انتخاب‌شده پشتیبانی نمی‌شود.", "ig_crawler_bad_provider")
    return newest(await _apify(cfg, username))


async def check_key(cfg: dict) -> tuple[bool, str]:
    """Admin "test" button: is the token valid? Costs nothing (reads the Apify account)."""
    if not cfg.get("token"):
        return False, "کلیدی ذخیره نشده است."
    try:
        async with httpx.AsyncClient(timeout=30, proxy=cfg.get("proxy") or None) as client:
            resp = await client.get("https://api.apify.com/v2/users/me", params={"token": cfg["token"]})
    except httpx.HTTPError:
        return False, "ارتباط با Apify برقرار نشد؛ اینترنت یا پروکسی را بررسی کن."
    if resp.status_code in (401, 403):
        return False, "کلید Apify نامعتبر است."
    if resp.status_code >= 400:
        return False, f"Apify خطا داد ({resp.status_code})."
    name = (resp.json().get("data") or {}).get("username", "")
    return True, f"کلید سالم است{f' (حساب {name})' if name else ''}؛ کرال اینستاگرام برای همهٔ فروشنده‌ها باز است."


def newest(posts: list[IgPost]) -> list[IgPost]:
    """The newest LIMIT posts by date (pinned posts come first in the feed even when they are old)."""
    return sorted(posts, key=lambda p: p.taken_at, reverse=True)[:LIMIT]


async def _apify(cfg: dict, username: str) -> list[IgPost]:
    url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync-get-dataset-items"
    body = {"directUrls": [f"https://www.instagram.com/{username}/"], "resultsType": "posts",
            "resultsLimit": LIMIT, "addParentData": False}
    try:
        async with httpx.AsyncClient(timeout=300, proxy=cfg.get("proxy") or None) as client:
            resp = await client.post(url, params={"token": cfg["token"]}, json=body)
    except httpx.HTTPError as exc:
        raise PlatformError("ارتباط با سرویس کرال اینستاگرام (Apify) برقرار نشد؛ اینترنت یا پروکسی را بررسی کن.",
                            "ig_crawler_network", detail=str(exc)[:300]) from exc
    if resp.status_code in (401, 403):
        raise PlatformError("کلید Apify نامعتبر است یا دسترسی ندارد.", "ig_crawler_bad_key", detail=resp.text[:300])
    if resp.status_code == 402:
        raise PlatformError("اعتبار حساب Apify تمام شده است.", "ig_crawler_no_credit", detail=resp.text[:300])
    if resp.status_code >= 400:
        raise PlatformError("سرویس کرال اینستاگرام خطا داد.", f"ig_crawler_http_{resp.status_code}", detail=resp.text[:300])
    rows = resp.json()
    if isinstance(rows, list) and rows and all(r.get("error") for r in rows if isinstance(r, dict)):
        # Apify returns one {"error": "not_found"|"private"…} row for a missing or private page.
        raise PlatformError("این پیج اینستاگرام پیدا نشد یا خصوصی است.", "ig_profile_unavailable",
                            detail=str(rows[0])[:300])
    return [p for p in (parse_apify(r) for r in rows if isinstance(r, dict)) if p]


def parse_apify(r: dict) -> IgPost | None:
    code = r.get("shortCode") or r.get("shortcode") or ""
    if not code or r.get("error"):
        return None
    typ = (r.get("type") or "").lower()
    is_reel = typ == "video" or r.get("productType") in ("clips", "igtv") or bool(r.get("videoUrl"))
    if is_reel:
        kind, images = "reel", [r.get("displayUrl")] if r.get("displayUrl") else []
    elif typ == "sidecar" or r.get("childPosts") or len(r.get("images") or []) > 1:
        kind = "carousel"
        children = [c.get("displayUrl") for c in r.get("childPosts") or [] if (c.get("type") or "").lower() != "video"]
        images = [u for u in (children or r.get("images") or [r.get("displayUrl")]) if u]
    else:
        kind, images = "image", [r.get("displayUrl")] if r.get("displayUrl") else []
    ts = r.get("timestamp") or ""
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts, timezone.utc).isoformat()
    return IgPost(shortcode=code, kind=kind, caption=r.get("caption") or "", url=r.get("url") or f"https://www.instagram.com/p/{code}/",
                  image_urls=images[:10], taken_at=str(ts))


async def download(url: str, proxy: str = "") -> tuple[bytes, str]:
    """Instagram CDN links expire within days, so photos are copied to our media folder."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, proxy=proxy or None) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    if len(resp.content) > 10 * 1024 * 1024:
        raise ValueError("image too large")
    return resp.content, resp.headers.get("content-type", "image/jpeg").split(";")[0]
