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
            return {"data": {"id": 55}}
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
