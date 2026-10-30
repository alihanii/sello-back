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
        return "**Keep the exact white ceramic mug; do NOT keep the table or shadows.**", 10, 5

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
    assert [x["name"] for x in op2["steps"]][-2:] == ["image_edit", "image_retry"]  # one simpler retry
    assert "IMAGE_SAFETY" in str(op2["steps"][-2]["response"])  # the reason stays in the log

    # no photo: a product with only text gets a generated photo, styled after the shop's earlier product photos
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
    assert builder["request"]["has_photo"] is False and builder["request"]["style_refs"] >= 1
    assert calls[-1][1] is False and calls[-1][2] == builder["request"]["style_refs"]  # only style photos were sent
    assert b"style references" in sent["body"]
    prod3 = (await client.get(f"/products/{op3['result']['product_id']}", headers=h)).json()
    assert prod3["images"][0]["url"] == op3["result"]["clean_image"]

    # seller-picked style samples replace the automatic ones
    r = await client.post(f"/products/{prod3['id']}/clean-image", headers=h,
                          json={"style_product_ids": [prod["id"]], "style_images": ["/media/999/x.png"]})
    await drain()
    op5 = (await client.get(f"/operations/{r.json()['id']}", headers=h)).json()
    assert op5["status"] == "succeeded", op5
    step = next(s for s in op5["steps"] if s["name"] == "image_prompt")
    assert step["request"]["style_refs"] == 1  # the chosen product's photo; the other store's path is ignored

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
                          has_photo=True, styles=0, design=True)
    assert "NOT the product" in text and "printed on the chest" in text
    assert "Analyze the attached e-commerce image" not in text
    plain = _builder_input(title="هودی", note="", category="general", has_photo=True, styles=0)
    assert "is the product photo" in plain
