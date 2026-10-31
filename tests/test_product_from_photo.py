import base64

import httpx

from app.services import ai
from tests.conftest import drain
from tests.conftest import login as _login

_tokens: dict = {}


async def login(client, phone):  # the admin phone hits the OTP rate limit if every test logs in again
    if phone not in _tokens:
        _tokens[phone] = await _login(client, phone)
    return _tokens[phone]

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def test_model_kinds():
    assert ai.model_kind("gemini-3.6-flash")[0] == "text"
    assert ai.model_kind("gemini-3.8-flash-tts")[0] == "unsupported"
    assert ai.model_kind("gemini-embedding-2")[0] == "unsupported"
    assert ai.model_kind("gemini-3.1-flash-image")[0] == "image"
    assert ai.check_model("product_generate", "gemini-3.1-flash-image")
    assert ai.check_model("image_edit", "gemini-3.6-flash")
    assert not ai.check_model("image_edit", "gemini-3.1-flash-image")
    assert not ai.check_model("vision", "gemini-3.6-flash")


async def test_admin_cannot_pick_wrong_models(client):
    admin = await login(client, "09120000000")  # noqa
    r = await client.post("/admin/ai/providers", headers=admin, json={
        "name": "tts", "kind": "gemini", "api_key": "k-1", "default_model": "gemini-3.8-flash-tts"})
    assert r.status_code == 422 and "صدا" in r.json()["detail"]["message"]
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "g", "kind": "gemini", "api_key": "k-2", "default_model": "gemini-3.6-flash"})).json()
    r = await client.put("/admin/ai/routes/image_edit", headers=admin, json={"provider_id": prov["id"], "model": ""})
    assert r.status_code == 422
    r = await client.put("/admin/ai/routes/image_edit", headers=admin, json={"provider_id": prov["id"], "model": "gemini-3.6-flash"})
    assert r.status_code == 422
    r = await client.put("/admin/ai/routes/product_generate", headers=admin, json={"provider_id": prov["id"], "model": "gemini-3.1-flash-image"})
    assert r.status_code == 422


async def test_raw_photo_becomes_product_with_clean_main_image(client, monkeypatch):
    admin = await login(client, "09120000000")
    h = await login(client, "09125555555")
    sid = (await client.post("/stores", json={"name": "عکس"}, headers=h)).json()["id"]
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "g2", "kind": "gemini", "api_key": "k-3", "default_model": "gemini-3.6-flash"})).json()
    for task in ("product_generate", "vision"):
        assert (await client.put(f"/admin/ai/routes/{task}", headers=admin, json={"provider_id": prov["id"]})).status_code == 200

    up = (await client.post(f"/stores/{sid}/uploads", headers=h, files={"files": ("raw.png", PNG, "image/png")})).json()["urls"]
    # no image model yet: asking for a clean photo is refused with the reason
    r = await client.post(f"/stores/{sid}/reference-sets", headers=h, json={"images": up, "note": "ماگ سرامیکی سفید", "clean_image": True})
    assert r.status_code == 422 and "تصویری" in r.json()["detail"]["message"]
    assert (await client.put("/admin/ai/routes/image_edit", headers=admin,
                             json={"provider_id": prov["id"], "model": "gemini-3.1-flash-image"})).status_code == 200
    assert (await client.get("/ai/status", headers=h)).json()["image_edit"] is True

    calls = []

    async def fake_call(provider, model, system, prompt, images, temperature, json_mode=True):
        calls.append((model, json_mode, len(images)))
        if json_mode:
            return ('{"title":"ماگ سرامیکی سفید","description":"ماگ سرامیکی ساده","category_hint":"ماگ",'
                    '"attributes":{"جنس":"سرامیک"},"image_category":"general"}'), 10, 5
        return "**Keep the exact white ceramic mug; do NOT keep the table or shadows. clean e-commerce cutout style.**", 10, 5

    sent = {}

    def handler(req):
        sent["url"], sent["body"] = str(req.url), req.content
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}]}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(ai, "_call", fake_call)
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))
    ops = (await client.post(f"/stores/{sid}/reference-sets", headers=h,
                             json={"images": up, "note": "ماگ سرامیکی سفید", "clean_image": True})).json()
    await drain()
    op = (await client.get(f"/operations/{ops[0]['id']}", headers=h)).json()
    assert op["status"] == "succeeded", op
    assert [s["name"] for s in op["steps"]][-2:] == ["image_prompt", "image_edit"]
    assert calls == [("gemini-3.6-flash", True, 1), ("gemini-3.6-flash", False, 1)]
    assert "gemini-3.1-flash-image:generateContent" in sent["url"] and b'"responseModalities":["IMAGE"]' in sent["body"]
    assert b"clean e-commerce cutout style." in sent["body"]
    prod = (await client.get(f"/products/{op['result']['product_id']}", headers=h)).json()
    assert prod["title"] == "ماگ سرامیکی سفید"
    assert prod["images"][0]["url"] == op["result"]["clean_image"] and prod["images"][1]["url"] == up[0]

    # the image model failing keeps the product and says why
    def broken(req):
        return httpx.Response(200, json={"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"parts": []}}]})

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(broken),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))
    r = await client.post(f"/products/{prod['id']}/clean-image", headers=h, json={"image_id": prod["images"][1]["id"]})
    assert r.status_code == 202
    await drain()
    op2 = (await client.get(f"/operations/{r.json()['id']}", headers=h)).json()
    assert op2["status"] == "failed" and "عکس دیگر" in op2["error_message_fa"]
    assert [x["name"] for x in op2["steps"]][-3:] == ["image_edit", "image_people", "image_retry"]  # one simpler retry
    assert "IMAGE_SAFETY" in str(op2["steps"][-3]["response"])  # the reason stays in the log

    # no photo: a product with only text gets a photo generated from the text; other products' photos are never sent
    calls.clear()

    def ok(req):
        sent["url"], sent["body"] = str(req.url), req.content
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}]}}]})

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(ok),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))
    ops = (await client.post(f"/stores/{sid}/reference-sets", headers=h,
                             json={"note": "ماگ سرامیکی مشکی", "clean_image": True})).json()
    await drain()
    op3 = (await client.get(f"/operations/{ops[0]['id']}", headers=h)).json()
    assert op3["status"] == "succeeded", op3
    builder = next(s for s in op3["steps"] if s["name"] == "image_prompt")
    assert builder["request"]["has_photo"] is False
    assert calls[-1][1] is False and calls[-1][2] == 0  # no images at all
    assert b"inlineData" not in sent["body"]
    prod3 = (await client.get(f"/products/{op3['result']['product_id']}", headers=h)).json()
    assert prod3["images"][0]["url"] == op3["result"]["clean_image"]

    # free tier with no image quota: after Google says so once, the app stops calling until the admin re-saves
    def free(req):
        return httpx.Response(429, json={"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0"}})

    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(free),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))
    r = await client.post(f"/products/{prod3['id']}/clean-image", headers=h, json={})
    await drain()
    op4 = (await client.get(f"/operations/{r.json()['id']}", headers=h)).json()
    assert op4["status"] == "failed" and "Billing" in op4["error_message_fa"]
    status = (await client.get("/ai/status", headers=h)).json()
    assert status["image_edit"] is False and status["image_edit_problem"] == ""  # sellers get no technical details
    assert "رایگان" in (await client.get("/ai/status", headers=admin)).json()["image_edit_problem"]
    r = await client.post(f"/products/{prod3['id']}/clean-image", headers=h, json={})
    assert r.status_code == 422
    assert (await client.put("/admin/ai/routes/image_edit", headers=admin,
                             json={"provider_id": prov["id"], "model": "gemini-3.1-flash-image"})).status_code == 200
    assert (await client.get("/ai/status", headers=h)).json()["image_edit"] is True


def test_design_photo_builds_product_around_the_artwork():
    from app.services.product_images import _builder_input

    text = _builder_input(title="هودی مشکی", note="هودی مشکی با طرح، سایز XXL", category="flat_clothing",
                          has_photo=True, design=True)
    assert "NOT the product" in text and "printed on the chest" in text
    assert "Analyze the attached e-commerce image" not in text
    plain = _builder_input(title="هودی", note="", category="general", has_photo=True)
    assert plain.startswith("PRODUCT NAME: هودی\nuser description: هودی\n")


async def test_design_flow_reaches_the_image_model(monkeypatch):
    """The design path builds its own system prompt; run make_photo end to end with fakes."""
    from app.services import product_images
    from app.services.media import save_bytes

    url = save_bytes(1, PNG, content_type="image/png")
    seen = {}

    async def fake_text(**kw):
        return "A black hoodie with the design on the chest."

    async def fake_image(*, system, prompt, images, store_id, task="image_edit"):
        seen.update(system=system, images=len(images))
        return PNG, "image/png"

    class Ctx:
        db = None

        def step(self, *a, **kw):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def cm():
                class S:
                    response = {}
                yield S()
            return cm()

    monkeypatch.setattr(product_images.ai, "generate_text", fake_text)
    monkeypatch.setattr(product_images.ai, "generate_image", fake_image)
    out = await product_images.make_photo(Ctx(), store_id=1, title="هودی", note="هودی مشکی", image_url=url, design=True)
    assert out.startswith("/media/1/") and seen["images"] == 1 and "DESIGN" in seen["system"]


def test_faces_are_blanked_and_photo_cropped_to_the_product():
    import io

    from PIL import Image

    from app.services.product_images import crop_without_faces

    img = Image.new("RGB", (100, 200), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out, mime = crop_without_faces(buf.getvalue(), [250, 0, 1000, 1000], [[0, 300, 200, 700]])
    res = Image.open(io.BytesIO(out))
    assert mime == "image/png" and res.size[1] < 200  # cropped below the top
    nested, _ = crop_without_faces(buf.getvalue(), [[250, 0, 1000, 1000]], [])
    assert Image.open(io.BytesIO(nested)).size[1] < 200
    assert crop_without_faces(buf.getvalue(), None, []) is None  # nothing to hide
    blank, _ = crop_without_faces(buf.getvalue(), None, [[0, 300, 200, 700]])
    assert Image.open(io.BytesIO(blank)).getpixel((50, 20)) == (255, 255, 255)


async def test_worn_clothing_hides_the_person_before_the_image_model(monkeypatch):
    import io

    from PIL import Image

    from app.services import product_images
    from app.services.media import save_bytes

    buf = io.BytesIO()
    Image.new("RGB", (100, 200), (0, 0, 0)).save(buf, "PNG")
    url = save_bytes(1, buf.getvalue(), content_type="image/png")
    seen = {}

    async def fake_json(**kw):
        return {"people": True, "product_box": [250, 0, 1000, 1000], "faces": [[0, 300, 200, 700]]}

    async def fake_text(**kw):
        seen["builder"] = kw["prompt"]
        return "The olive hoodie alone as a ghost mannequin."

    async def fake_image(*, system, prompt, images, store_id, task="image_edit"):
        seen["sent"] = images[0][0]
        return PNG, "image/png"

    async def no_route(task):
        return False

    class Ctx:
        def step(self, *a, **kw):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def cm():
                class S:
                    response = {}
                yield S()
            return cm()

    monkeypatch.setattr(product_images.ai, "generate_json", fake_json)
    monkeypatch.setattr(product_images.ai, "generate_text", fake_text)
    monkeypatch.setattr(product_images.ai, "generate_image", fake_image)
    monkeypatch.setattr(product_images.ai, "has_route", no_route)
    await product_images.make_photo(Ctx(), store_id=1, title="هودی", note="", category="modeled_clothing", image_url=url)
    assert "ghost mannequin" in seen["builder"] and "Flat Lay" in seen["builder"]
    assert Image.open(io.BytesIO(seen["sent"])).size[1] < 200  # the cropped photo, not the original
    # any other photo goes to the builder exactly as platform-wrappers sends it
    await product_images.make_photo(Ctx(), store_id=1, title="هودی", note="هودی سبز", category="flat_clothing", image_url=url)
    assert seen["builder"] == ("PRODUCT NAME: هودی سبز\nuser description: هودی سبز\nSelected category (prompt_type 1): "
                               "General Clothing (Apparel - Flat Lay)\n\n"
                               "Analyze the attached e-commerce image and return exactly one raw English paragraph.")
    assert seen["sent"] == buf.getvalue()


async def test_openrouter_key_makes_images(client, monkeypatch):
    """OpenRouter: chat/completions with image modality, the picture comes back as a data URL."""
    admin = await login(client, "09120000000")
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "or", "kind": "openrouter", "api_key": "k-or", "default_model": "google/gemini-3.6-flash"})).json()
    r = await client.put("/admin/ai/routes/image_edit", headers=admin,
                         json={"provider_id": prov["id"], "model": "google/gemini-3.1-flash-lite-image"})
    assert r.status_code == 200, r.text
    sent = {}

    def handler(req):
        sent["url"], sent["body"] = str(req.url), req.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "images": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}})

    real = httpx.AsyncClient
    monkeypatch.setattr(ai.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler),
                                                                      **{k: v for k, v in kw.items() if k != "proxy"}))
    data, mime = await ai.generate_image(system="s", prompt="p", images=[(PNG, "image/png")])
    assert data == PNG and mime == "image/png"
    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert b'"modalities":["image","text"]' in sent["body"] and b"google/gemini-3.1-flash-lite-image" in sent["body"]
    assert b"data:image/png;base64," in sent["body"]  # the seller's photo goes along
