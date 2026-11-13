import io
from contextlib import asynccontextmanager
from types import SimpleNamespace

from PIL import Image

from app.platforms import digikala as dk


class Ctx:
    def __init__(self):
        self.steps = []
        self.db = None

    @asynccontextmanager
    async def step(self, key, title, request=None):
        s = SimpleNamespace(response=None)
        self.steps.append(key)
        yield s


def png_url(tmp_path, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "media_dir", str(tmp_path))
    buf = io.BytesIO()
    Image.new("RGBA", (40, 40), (200, 0, 0, 128)).save(buf, "PNG")
    (tmp_path / "a.png").write_bytes(buf.getvalue())
    return "/media/a.png"


async def test_digikala_publish_runs_whole_wizard(tmp_path, monkeypatch):
    calls = []
    uploads = []

    async def fake_request(self, method, path, *, token=None, body=None, referer=None, files=None, data=None):
        calls.append((method, path.split("?")[0], body, files))
        if "/search/category/" in path:
            return {"data": {"items": [{"id": 11, "text": "تیشرت مردانه", "leaf": True}]}}
        if path.endswith("/validation") and "category" in path:
            return {"data": {"is_valid": True, "bind": {
                "brands": [{"id": 719, "title_fa": "متفرقه"}], "divisions": [{"value": 5, "text": "پوشاک"}],
                "general_mefa": {"a": {"value": 9, "text": "عمومی", "general_id": "272"}}}}}
        if path.endswith("/detail/validation"):
            return {"data": {"draft_product_id": 777}}
        if path.startswith("/api/v2/product-creation/attributes/"):
            return {"data": {"category_group_attributes": {"g": {"attributes": {
                "1": {"id": 1, "title": "جنس", "type": "select", "required": True,
                      "values": {"31": {"text": "پنبه"}, "32": {"text": "پلی‌استر"}}},
                "2": {"id": 2, "title": "وزن", "type": "number", "required": False}}}}}}
        if path.endswith("/temp_images"):
            assert files["file"][1][:2] == b"\xff\xd8"  # re-encoded to JPEG
            img = Image.open(io.BytesIO(files["file"][1]))
            assert img.size == (1200, 1200) and img.getpixel((2, 2)) == (255, 255, 255)  # square, white, Digikala size
            uploads.append(files["file"][1])
            return {"status": "ok", "data": {}}  # an answer without the id: the draft's own list is read instead
        if "/temp_images/draft_product/" in path:
            return {"data": {"items": [{"id": 55 + n, "url": f"https://x/{55 + n}.jpg", "is_main": False}
                                       for n in range(len(uploads))]}}
        if path.endswith("/save") and "auto-title" not in path:
            return {"data": {"product_id": 123456}}
        return {"data": {}}

    async def no_ai(task="x"):
        return False

    monkeypatch.setattr(dk.DigikalaAdapter, "_request", fake_request)
    monkeypatch.setattr(dk.DigikalaAdapter, "_token", lambda self, ch: "tok")
    monkeypatch.setattr(dk.ai, "ai_available", no_ai)
    product = SimpleNamespace(
        title="تیشرت مردانه نخی", description="", brand="", category_hint="تیشرت مردانه", sku="", store_id=1,
        attributes={"جنس": "پنبه"}, images=[SimpleNamespace(url=png_url(tmp_path, monkeypatch))])
    ctx = Ctx()
    res = await dk.digikala.publish(ctx, SimpleNamespace(), product, SimpleNamespace())

    assert res.remote_id == "DKP-123456"
    attrs = next(b for m, p, b, f in calls if p == "/api/v2/product-creation/attributes")
    assert attrs["attributes"] == [{"id": 1, "value": 31}] and attrs["draft_product_id"] == 777
    draft = next(b for m, p, b, f in calls if p.endswith("/detail/validation"))
    assert draft["brand_id"] == 719 and draft["division_id"] == 5 and draft["general_mefa_id"] == 9
    assert ctx.steps[-3:] == ["images", "save", "submit"]
    assert ("PATCH", "/api/v2/product-creation/temp_images/55/main", None, None) in calls
    # One photo: Digikala needs two, so a wider-framed copy of the first goes too (not the same file twice).
    assert len(uploads) == 2 and uploads[0] != uploads[1]


async def test_main_image_made_once_with_the_smart_product_pipeline(monkeypatch):
    made = []

    async def fake_make_photo(ctx, **kw):
        made.append(kw)
        return "/media/1/clean.png"

    async def yes(task="x"):
        return True

    import app.services.product_images as pi
    monkeypatch.setattr(pi, "make_photo", fake_make_photo)
    monkeypatch.setattr(dk.ai, "ai_available", yes)
    product = SimpleNamespace(title="هودی", description="هودی مشکی", store_id=1, source_ref={},
                              images=[SimpleNamespace(url="/media/1/raw.jpg", origin_url="")])
    ctx = Ctx()
    assert await dk.digikala._main_image(ctx, product, "") == "/media/1/clean.png"
    assert await dk.digikala._main_image(ctx, product, "") == "/media/1/clean.png"  # reused on a retry
    assert len(made) == 1 and made[0]["image_url"] == "/media/1/raw.jpg"
    product.images[0].origin_url = "/media/1/orig.jpg"  # already cleaned by smart product creation
    product.source_ref = {}
    assert await dk.digikala._main_image(ctx, product, "") == ""
