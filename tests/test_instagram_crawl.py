import httpx

from app.services import instagram_crawl as ig
from tests.conftest import drain
from tests.conftest import login as _login

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
_tokens: dict = {}


async def login(client, phone):
    if phone not in _tokens:
        _tokens[phone] = await _login(client, phone)
    return _tokens[phone]


def test_usernames_and_post_kinds():
    assert ig.normalize_username("@My.Shop") == "my.shop"
    assert ig.normalize_username("https://www.instagram.com/my_shop/?hl=fa") == "my_shop"
    assert ig.normalize_username("https://instagram.com/p/ABC/") == ""
    assert ig.normalize_username("not a name") == ""
    reel = ig.parse_apify({"shortCode": "R1", "type": "Video", "displayUrl": "https://cdn/r.jpg", "videoUrl": "https://cdn/r.mp4"})
    assert reel.kind == "reel" and reel.image_urls == ["https://cdn/r.jpg"]  # only the cover
    car = ig.parse_apify({"shortCode": "C1", "type": "Sidecar", "displayUrl": "https://cdn/a.jpg", "childPosts": [
        {"type": "Image", "displayUrl": "https://cdn/a.jpg"}, {"type": "Video", "displayUrl": "https://cdn/v.jpg"},
        {"type": "Image", "displayUrl": "https://cdn/b.jpg"}]})
    assert car.kind == "carousel" and car.image_urls == ["https://cdn/a.jpg", "https://cdn/b.jpg"]
    # pinned old posts come first in the feed; only the newest 25 by date are kept
    posts = [ig.IgPost(shortcode=f"P{i}", kind="image", taken_at=f"2026-10-{i + 1:02d}T00:00:00Z") for i in range(30)]
    kept = ig.newest(list(reversed(posts)))
    assert len(kept) == 25 and kept[0].shortcode == "P29" and "P4" not in {p.shortcode for p in kept}


async def test_instagram_page_crawl(client, monkeypatch):
    admin = await login(client, "09120000000")
    h = await login(client, "09126666666")
    sid = (await client.post("/stores", json={"name": "ig"}, headers=h)).json()["id"]

    # locked until the admin sets the crawl key
    caps = (await client.get("/crawl/capabilities", headers=h)).json()["instagram_profile"]
    assert caps["available"] is False and "ادمین" in caps["reason"]
    igp = next(p for p in (await client.get("/platforms", headers=h)).json() if p["platform"] == "instagram")
    assert igp["enabled"] is False and igp["crawl_only"] is False
    r = await client.post(f"/stores/{sid}/crawl-sources", headers=h, json={"type": "instagram_profile", "url": "@shoe.shop"})
    assert r.status_code == 422 and "ادمین" in r.json()["detail"]["message"]

    cfg = (await client.put("/admin/integrations/instagram-crawler", headers=admin,
                            json={"provider": "apify", "token": "apify_api_test_1234"})).json()
    assert cfg["has_token"] and "token" not in cfg and cfg["token_hint"] == "…1234"
    assert (await client.get("/crawl/capabilities", headers=h)).json()["instagram_profile"]["available"] is True
    igp = next(p for p in (await client.get("/platforms", headers=h)).json() if p["platform"] == "instagram")
    assert igp["enabled"] and igp["crawl_only"] and igp["crawl_ready"] and not igp["publish_ready"]

    r = await client.post(f"/stores/{sid}/crawl-sources", headers=h, json={"type": "instagram_profile", "url": "bad name"})
    assert r.status_code == 422
    src = (await client.post(f"/stores/{sid}/crawl-sources", headers=h, json={
        "type": "instagram_profile", "url": "@Shoe.Shop", "extraction_mode": "regex"})).json()
    assert src["url"] == "shoe.shop" and src["include_reels"] is False

    rows = [
        {"shortCode": "A1", "type": "Image", "caption": "کفش اسپرت مردانه\nقیمت: ۲۵۰ هزار تومان", "displayUrl": "https://cdn/a1.jpg",
         "timestamp": "2026-10-30T10:00:00.000Z"},
        {"shortCode": "C1", "type": "Sidecar", "caption": "کفش کتانی زنانه\nقیمت: ۳۰۰ هزار تومان", "displayUrl": "https://cdn/c1.jpg",
         "childPosts": [{"type": "Image", "displayUrl": "https://cdn/c1.jpg"}, {"type": "Image", "displayUrl": "https://cdn/c2.jpg"}],
         "timestamp": "2026-10-29T10:00:00.000Z"},
        {"shortCode": "R1", "type": "Video", "productType": "clips", "caption": "کفش رسمی\nقیمت: ۴۰۰ هزار تومان",
         "displayUrl": "https://cdn/r1.jpg", "videoUrl": "https://cdn/r1.mp4", "timestamp": "2026-10-28T10:00:00.000Z"},
    ]
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if "api.apify.com" in req.url.host:
            assert req.url.params["token"] == "apify_api_test_1234"
            assert b'"resultsLimit":25' in req.content and b"instagram.com/shoe.shop/" in req.content
            return httpx.Response(201, json=rows)
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    real = httpx.AsyncClient
    monkeypatch.setattr(ig.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))

    async def crawl():
        o = (await client.post(f"/crawl-sources/{src['id']}/run", headers=h)).json()
        await drain()
        return (await client.get(f"/operations/{o['id']}", headers=h)).json()

    first = await crawl()
    assert first["status"] == "succeeded", first
    assert first["result"]["created"] == 2 and first["result"]["reels_skipped"] == 1 and first["result"]["seen"] == 3
    assert "ریلز" in first["result"]["note"]
    prods = (await client.get(f"/stores/{sid}/products", headers=h)).json()
    items = prods["items"] if isinstance(prods, dict) else prods
    by_title = {p["title"]: p for p in items}
    shoe = by_title["کفش کتانی زنانه"]
    assert len(shoe["images"]) == 2 and all(i["url"].startswith("/media/") for i in shoe["images"])  # copied, not linked

    # the same posts again: nothing is imported twice
    second = await crawl()
    assert second["result"]["created"] == 0 and second["result"]["not_loaded"] == 2
    assert "قبلاً کرال" in second["result"]["note"]

    # reels on: only the reel is new, and only its cover image is taken
    r = await client.patch(f"/crawl-sources/{src['id']}", headers=h, json={**src, "include_reels": True})
    assert r.json()["include_reels"] is True
    third = await crawl()
    assert third["result"]["created"] == 1 and third["result"]["not_loaded"] == 2
    prods = (await client.get(f"/stores/{sid}/products", headers=h)).json()
    items = prods["items"] if isinstance(prods, dict) else prods
    reel = next(p for p in items if p["title"] == "کفش رسمی")
    assert len(reel["images"]) == 1
    assert not any(u.endswith(".mp4") for u in calls)
