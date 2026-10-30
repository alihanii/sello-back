import io
from types import SimpleNamespace

from PIL import Image

from app.core.config import settings
from app.core.security import encrypt_credentials
from app.platforms import instagram as ig
from tests.test_digikala_publish import Ctx

APP = {"enabled": True, "app_id": "1", "app_secret": "s", "proxy": "", "oauth_ready": True,
       "redirect_uri": "http://localhost:3000/app/oauth/instagram"}


async def fake_app(db):
    return APP


async def test_instagram_oauth_connect_and_carousel_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_dir", str(tmp_path))
    monkeypatch.setattr(settings, "public_base_url", "https://api.sello.test")
    monkeypatch.setattr(ig, "instagram_app", fake_app)
    monkeypatch.setattr(ig.InstagramAdapter, "poll_seconds", 0)
    for n in ("a.png", "b.webp"):
        buf = io.BytesIO()
        Image.new("RGB", (30, 30), "blue").save(buf, "PNG" if n.endswith("png") else "WEBP")
        (tmp_path / n).write_bytes(buf.getvalue())
    calls, polls = [], {}

    async def fake_http(self, app, method, url, **kw):
        body = kw.get("data") or kw.get("params") or {}
        calls.append((method, url.split("/v25.0/")[-1], body))
        if url == ig.OAUTH_TOKEN:
            return {"access_token": "short", "user_id": 42}
        if url.endswith("/access_token"):
            return {"access_token": "long", "expires_in": 5184000}
        if url.endswith("/me"):
            return {"user_id": "1784", "username": "shop", "account_type": "BUSINESS"}
        if url.endswith("content_publishing_limit"):
            return {"data": [{"quota_usage": 3, "config": {"quota_total": 100}}]}
        if url.endswith("/media") and method == "POST":
            assert body["access_token"] == "long"
            return {"id": f"c{len([c for c in calls if c[1].endswith('/media')])}"}
        if url.endswith("media_publish"):
            return {"id": "m1"}
        if body.get("fields") == "status_code,status":
            cid = url.rsplit("/", 1)[-1]
            polls[cid] = polls.get(cid, 0) + 1
            return {"status_code": "FINISHED" if polls[cid] > 1 else "IN_PROGRESS"}
        if body.get("fields") == "permalink":
            return {"permalink": "https://instagram.com/p/x"}
        raise AssertionError(url)

    monkeypatch.setattr(ig.InstagramAdapter, "_http", fake_http)
    ctx = Ctx()
    res = await ig.instagram.connect(ctx, SimpleNamespace(), {"code": "abc#_"})
    assert res["config"]["ig_user_id"] == "1784" and res["credentials"]["access_token"] == "long"
    assert calls[0][2]["code"] == "abc"

    channel = SimpleNamespace(credentials_encrypted=encrypt_credentials(res["credentials"]), config=res["config"])
    product = SimpleNamespace(title="کیف چرم", description="دست‌دوز", price=0, compare_at_price=0, post_caption="",
                              stock_qty=1, attributes={}, tags=[],
                              images=[SimpleNamespace(url="/media/a.png"), SimpleNamespace(url="/media/b.webp")])
    out = await ig.instagram.publish(ctx, channel, product, SimpleNamespace(settings={}))
    assert out.remote_id == "m1" and out.remote_url.endswith("/p/x")
    media = [c for c in calls if c[1].endswith("/media")]
    assert media[0][2]["image_url"].startswith("https://api.sello.test/media/out/") and media[0][2]["image_url"].endswith(".jpg")
    assert media[-1][2]["media_type"] == "CAROUSEL" and media[-1][2]["children"] == "c1,c2"
    assert list((tmp_path / "out").iterdir())[0].read_bytes()[:2] == b"\xff\xd8"
