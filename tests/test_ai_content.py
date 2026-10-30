from app.platforms.botapi import BotApiAdapter
from tests.conftest import drain, login
from tests.test_flow import fake_call


async def test_ai_content_automation_needs_approval_then_publishes(client, monkeypatch):
    monkeypatch.setattr(BotApiAdapter, "_call", fake_call)
    admin = await login(client, "09120000000")
    h = await login(client, "09124444444")
    sid = (await client.post("/stores", json={"name": "محتوا"}, headers=h)).json()["id"]
    await client.post(f"/stores/{sid}/channels/connect", headers=h,
                      json={"platform": "telegram", "data": {"bot_token": "123:abc", "chat_id": "@testshop"}})
    await drain()
    cid = (await client.get(f"/stores/{sid}/channels", headers=h)).json()[0]["id"]
    pid = (await client.post(f"/stores/{sid}/products", headers=h, json={"title": "کیف چرم", "price": 1_200_000})).json()["id"]
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    up = (await client.post(f"/stores/{sid}/uploads", headers=h, files={"files": ("a.png", png, "image/png")})).json()
    await client.patch(f"/products/{pid}", headers=h, json={"images": up["urls"]})

    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "k", "kind": "gemini", "api_key": "sk-1", "default_model": "gemini-2.5-flash"})).json()
    await client.put("/admin/ai/routes/caption_rewrite", headers=admin, json={"provider_id": prov["id"], "model": "gemini-2.5-flash"})
    from app.services import ai

    async def fake(provider, model, system, prompt, images, temperature):
        assert "کیف چرم" in prompt
        return '{"caption":"کیف چرم دست‌دوز 👜\\nمناسب هر روز","hook":"کیف چرم","hashtags":["#کیف"]}', 100, 50

    monkeypatch.setattr(ai, "_call", fake)
    rule = (await client.post(f"/stores/{sid}/automations", headers=h, json={
        "name": "پست‌های هفتگی", "trigger": "ai_content",
        "actions": {"channel_ids": [cid], "weekdays": [0, 2, 4], "hour": 11, "source": "catalog", "brief": "لحن صمیمی"}})).json()
    ops = (await client.post(f"/automations/{rule['id']}/run", headers=h)).json()["operation_ids"]
    await drain()
    op = (await client.get(f"/operations/{ops[0]}", headers=h)).json()
    assert op["status"] == "succeeded", op

    # owner is notified and must approve; nothing was published yet
    notes = (await client.get(f"/notifications?store_id={sid}", headers=h)).json()
    assert notes["unread"] == 1 and notes["items"][0]["kind"] == "approval"
    pending = (await client.get(f"/stores/{sid}/approvals", headers=h)).json()
    assert pending[0]["caption"].startswith("کیف چرم دست‌دوز") and pending[0]["kind"] == "ai_post"
    assert (await client.get(f"/products/{pid}", headers=h)).json()["listings"] == []

    r = (await client.post(f"/approvals/{pending[0]['id']}", headers=h,
                           json={"approve": True, "caption": "کیف چرم دست‌دوز — ویرایش من"})).json()
    assert len(r["operation_ids"]) == 1
    await drain()
    listing = (await client.get(f"/products/{pid}", headers=h)).json()["listings"][0]
    assert listing["status"] == "published" and listing["published_by"] == "ai"
    log = (await client.get(f"/operations/{r['operation_ids'][0]}", headers=h)).json()
    caption = next(s for s in log["steps"] if s["name"] == "build_caption")["response"]["caption"]
    assert caption.startswith("کیف چرم دست‌دوز — ویرایش من")

    # nothing left to post → a clear Persian failure, plus a notification
    ops = (await client.post(f"/automations/{rule['id']}/run", headers=h)).json()["operation_ids"]
    await drain()
    op = (await client.get(f"/operations/{ops[0]}", headers=h)).json()
    assert op["status"] == "failed" and op["error_code"] == "nothing_to_post"

    # leave no AI key behind for the other tests sharing this database
    assert (await client.delete(f"/admin/ai/providers/{prov['id']}", headers=admin)).status_code == 200
