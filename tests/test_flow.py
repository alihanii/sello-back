from app.platforms.botapi import BotApiAdapter
from tests.conftest import drain, login


async def fake_call(self, token, method, data=None, files=None):
    if token == "bad":
        return (_ for _ in ()).throw(self._humanize({"ok": False, "error_code": 401, "description": "Unauthorized"}))
    return {
        "getMe": {"id": 1, "username": "sello_bot"},
        "getChat": {"id": -100123, "title": "فروشگاه تست", "username": "testshop"},
        "getChatMember": {"status": "administrator"},
        "sendPhoto": {"message_id": 55},
        "sendMessage": {"message_id": 56},
        "sendMediaGroup": [{"message_id": 57}],
        "getUpdates": [
            {"update_id": 10, "channel_post": {"message_id": 70, "chat": {"id": -100123},
                                               "text": "شال نخی\nقیمت: ۳۵۰,۰۰۰\nموجودی: ۵"}},
            {"update_id": 11, "channel_post": {"message_id": 71, "chat": {"id": -100123}, "text": "سلام دوستان"}},
        ],
    }[method]


async def test_full_seller_flow(client, monkeypatch):
    monkeypatch.setattr(BotApiAdapter, "_call", fake_call)
    h = await login(client, "09121111111")

    store = (await client.post("/stores", json={"name": "بوتیک نمونه"}, headers=h)).json()
    sid = store["id"]

    # connect telegram -> async operation with steps
    op = (await client.post(f"/stores/{sid}/channels/connect", headers=h,
                            json={"platform": "telegram", "data": {"bot_token": "123:abc", "chat_id": "@testshop"}})).json()
    assert op["status"] == "queued"
    await drain()
    detail = (await client.get(f"/operations/{op['id']}", headers=h)).json()
    assert detail["status"] == "succeeded", detail
    assert [s["name"] for s in detail["steps"]] == ["get_me", "get_chat", "check_admin"]
    channels = (await client.get(f"/stores/{sid}/channels", headers=h)).json()
    assert channels[0]["status"] == "connected" and channels[0]["display_name"] == "فروشگاه تست"
    cid = channels[0]["id"]

    # product + publish
    p = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "مانتو", "price": 900000, "stock_qty": 4})).json()
    assert p["source"] == "manual"
    ops = (await client.post(f"/products/{p['id']}/publish", headers=h, json={"channel_ids": [cid]})).json()
    # a second click while queued must not create a duplicate publish
    again = (await client.post(f"/products/{p['id']}/publish", headers=h, json={"channel_ids": [cid]})).json()
    assert len(ops) == 1 and again == []
    await drain()
    p = (await client.get(f"/products/{p['id']}", headers=h)).json()
    listing = p["listings"][0]
    assert listing["status"] == "published" and listing["published_by"] == "user"
    assert listing["remote_url"] == "https://t.me/testshop/56"
    log = (await client.get(f"/operations/{ops[0]['id']}", headers=h)).json()
    assert [s["name"] for s in log["steps"]] == ["validate", "build_caption", "send"]
    assert "123:abc" not in str(log)  # token never reaches the visible log

    # filters: published / not published in channel
    r = (await client.get(f"/stores/{sid}/products?published_in={cid}", headers=h)).json()
    assert r["total"] == 1
    r = (await client.get(f"/stores/{sid}/products?not_published_in={cid}", headers=h)).json()
    assert r["total"] == 0

    # crawl channel posts (regex) -> one product, one failed item, partial status
    src = (await client.post(f"/stores/{sid}/crawl-sources", headers=h,
                             json={"type": "channel_posts", "channel_id": cid, "extraction_mode": "regex"})).json()
    op = (await client.post(f"/crawl-sources/{src['id']}/run", headers=h)).json()
    await drain()
    op = (await client.get(f"/operations/{op['id']}", headers=h)).json()
    assert op["result"]["created"] == 1
    items = (await client.get(f"/operations/{op['id']}/crawl-items", headers=h)).json()
    assert sorted(i["result"] for i in items) == ["created", "skipped"]
    assert op["status"] == "succeeded"
    crawled = (await client.get(f"/stores/{sid}/products?source=crawl", headers=h)).json()["items"][0]
    assert crawled["title"] == "شال نخی" and crawled["price"] == 350000 and crawled["stock_qty"] == 5

    # calendar + dashboard reflect the activity
    from datetime import date
    today = date.today().isoformat()
    cal = (await client.get(f"/stores/{sid}/calendar?start={today}&end={today}", headers=h)).json()
    day = cal["days"][today]
    assert day["publish_ok"] == 1 and day["crawl_run"] == 1
    dash = (await client.get(f"/stores/{sid}/dashboard", headers=h)).json()
    assert dash["kpis"]["products"] == 2 and dash["kpis"]["published_any"] == 1


async def test_failures_are_explained(client, monkeypatch):
    monkeypatch.setattr(BotApiAdapter, "_call", fake_call)
    h = await login(client, "09122222222")
    sid = (await client.post("/stores", json={"name": "s2"}, headers=h)).json()["id"]
    op = (await client.post(f"/stores/{sid}/channels/connect", headers=h,
                            json={"platform": "telegram", "data": {"bot_token": "bad", "chat_id": "@x"}})).json()
    await drain()
    op = (await client.get(f"/operations/{op['id']}", headers=h)).json()
    assert op["status"] == "failed"
    assert op["error_message_fa"] == "توکن ربات نامعتبر است یا ربات حذف شده."
    assert op["suggested_action"]["type"] == "reconnect"

    # AI not configured -> clear message pointing at the admin
    ops = (await client.post(f"/stores/{sid}/reference-sets", headers=h, json={"note": "کفش اسپرت مردانه"})).json()
    await drain()
    op = (await client.get(f"/operations/{ops[0]['id']}", headers=h)).json()
    assert op["status"] == "failed" and op["error_code"] == "ai_not_configured"


async def test_admin_plans_and_ai_keys(client, monkeypatch):
    admin = await login(client, "09120000000")
    seller = await login(client, "09123333333")
    plans = (await client.get("/plans")).json()
    assert {p["duration_months"] for p in plans} == {1, 3, 12}
    pro3 = next(p for p in plans if p["tier"] == "pro" and p["duration_months"] == 3)
    assert (await client.post("/plan-requests", headers=seller, json={"plan_id": pro3["id"]})).status_code == 200
    reqs = (await client.get("/admin/plan-requests", headers=admin)).json()
    assert (await client.post(f"/admin/plan-requests/{reqs[0]['id']}", headers=admin, json={"approve": True})).status_code == 200
    sub = (await client.get("/me/subscription", headers=seller)).json()
    assert sub["subscription"]["plan"]["tier"] == "pro" and sub["limits"]["max_stores"] == 3
    assert (await client.get("/admin/stats", headers=seller)).status_code == 403

    # AI key + route + a generation call, mocked at the HTTP layer
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "متیس", "kind": "gemini", "base_url": "https://api.metisai.ir", "api_key": "sk-test-9999",
        "default_model": "gemini-2.5-flash", "input_price_per_m": 1_000_000, "output_price_per_m": 2_000_000})).json()
    assert prov["key_hint"] == "9999" and "api_key" not in prov
    await client.put("/admin/ai/routes/product_generate", headers=admin, json={"provider_id": prov["id"], "model": "gemini-2.5-flash"})

    from app.services import ai

    async def fake(provider, model, system, prompt, images, temperature):
        return ('{"title":"کفش اسپرت مردانه مدل رانر","description":"راحت و سبک","price":0,'
                '"attributes":{"جنس":"مش"},"tags":["کفش"],"field_sources":{"title":"note"}}', 1000, 500)

    monkeypatch.setattr(ai, "_call", fake)
    sid = (await client.post("/stores", json={"name": "s3"}, headers=seller)).json()["id"]
    ops = (await client.post(f"/stores/{sid}/reference-sets", headers=seller, json={"note": "کفش اسپرت", "price": 1500000})).json()
    await drain()
    op = (await client.get(f"/operations/{ops[0]['id']}", headers=seller)).json()
    assert op["status"] == "succeeded", op
    prod = (await client.get(f"/products/{op['result']['product_id']}", headers=seller)).json()
    assert prod["source"] == "ai" and prod["status"] == "draft" and prod["price"] == 1500000

    usage = (await client.get("/admin/ai/usage", headers=admin)).json()
    row = usage["providers"][0]
    assert row["calls"] == 1 and row["input_tokens"] == 1000 and row["cost_toman"] == 2000
    assert usage["by_task"][0]["task"] == "product_generate"
