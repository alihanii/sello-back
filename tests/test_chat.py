"""Sales chat assistant: connections, the cheap no-model paths, one model call per turn, orders, takeover, usage."""

import json

import pytest
from sqlalchemy import delete

from app.platforms.botapi import BotApiAdapter
from app.services import ai, chat_engine
from app.services import chat_runtime as rt
from tests.conftest import login as _login

_tokens: dict = {}


async def login(client, phone):
    if phone not in _tokens:
        _tokens[phone] = await _login(client, phone)
    return _tokens[phone]


SENT: list[tuple[str, dict]] = []


@pytest.fixture(autouse=True)
async def _no_ai_keys_left():
    """Other test files expect a fresh install without AI keys."""
    yield
    from app.core.db import SessionLocal
    from app.models import AiProvider, AiTaskRoute, AiUsageLog
    async with SessionLocal() as db:
        await db.execute(delete(AiUsageLog))
        await db.execute(delete(AiTaskRoute))
        await db.execute(delete(AiProvider))
        await db.commit()


async def fake_bot_call(self, token, method, data=None, files=None):
    SENT.append((method, data or {}))
    if token == "bad-token-000":
        raise self._humanize({"ok": False, "error_code": 401, "description": "Unauthorized"})
    return {
        "getMe": {"id": 900, "username": "shop_chat_bot", "first_name": "Shop", "can_connect_to_business": True},
        "deleteWebhook": True,
        "setWebhook": True,
        "sendMessage": {"message_id": len(SENT) + 1000},
        "sendPhoto": {"message_id": len(SENT) + 1000},
        "getBusinessConnection": {"id": "bc1", "user": {"id": 77, "first_name": "Ali"}, "user_chat_id": 77,
                                  "is_enabled": True, "rights": {"can_reply": True}},
    }[method]


class FakeLLM:
    """Scripted model: records every prompt so tests can check what (and how much) was sent."""

    def __init__(self):
        self.calls: list[dict] = []
        self.answers: list[dict] = []

    async def __call__(self, *, system, prompt, store_id, max_tokens=400, task="chat"):
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        out = self.answers.pop(0) if self.answers else {"reply": "بله موجوده", "intent": "product"}
        return out, {"model": "gemini-flash-lite", "input_tokens": 1200, "output_tokens": 60, "cached_tokens": 800,
                     "cost_toman": 3, "latency_ms": 420}


async def _setup(client, monkeypatch, phone="09125550001"):
    monkeypatch.setattr(BotApiAdapter, "_call", fake_bot_call)
    llm = FakeLLM()
    monkeypatch.setattr(chat_engine.ai, "generate_chat", llm)
    admin = await login(client, "09120000000")
    r = await client.post("/admin/ai/providers", headers=admin, json={
        "name": "chat-key", "kind": "gemini", "api_key": "k-chat", "default_model": "gemini-3.6-flash"})
    assert r.status_code in (200, 201), r.text
    h = await login(client, phone)
    sid = (await client.post("/stores", json={"name": "بوتیک چت"}, headers=h)).json()["id"]
    p1 = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "مانتو لینن کرم", "price": 1250000, "stock_qty": 3, "attributes": {"سایز": "۳۸ تا ۴۴"}})).json()
    p2 = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "شال نخی مشکی", "price": 350000, "stock_qty": 0})).json()
    return h, sid, llm, p1, p2


async def test_agent_settings_and_validation(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch)
    a = (await client.get(f"/stores/{sid}/chat/agent", headers=h)).json()
    assert a["enabled"] is False and "بوتیک چت" in a["defaults"]["welcome_message"]
    # orders need a way to pay
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"orders_enabled": True})
    assert r.status_code == 422
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "tone": "casual", "orders_enabled": True,
        "payment": {"card_number": "6037-0000-1111-2222", "holder": "علی", "require_receipt": True},
        "faqs": [{"q": "هزینه ارسال چقدره؟", "a": "ارسال با پست ۷۰ هزار تومان است."}, {"q": "", "a": "x"}],
        "business_hours": {"enabled": True, "days": {"sat": [["09:00", "21:00"]], "xyz": [["1", "2"]]}}})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["enabled"] and a["tone"] == "casual" and len(a["faqs"]) == 1
    assert list(a["business_hours"]["days"]) == ["sat"]
    assert (await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"tone": "rude"})).status_code == 422


async def test_connect_bot_and_business(client, monkeypatch):
    h, sid, *_ = await _setup(client, monkeypatch, "09125550002")
    r = await client.post(f"/stores/{sid}/chat/bots", headers=h, json={"platform": "bale", "mode": "business",
                                                                         "bot_token": "123456:abcdefgh"})
    assert r.status_code == 422 and "بله" in r.json()["detail"]["message"]
    r = await client.post(f"/stores/{sid}/chat/bots", headers=h, json={"platform": "telegram", "mode": "bot",
                                                                         "bot_token": "bad-token-000"})
    assert r.status_code == 400
    bot = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "111:chat-token-a"})).json()
    assert bot["status"] == "connected" and bot["bot_username"] == "shop_chat_bot"
    assert "111:chat-token-a" not in json.dumps(bot)
    # same token twice is refused (both would read getUpdates)
    r = await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "111:chat-token-a"})
    assert r.status_code == 422

    # a bot already used to publish in a channel is refused too
    async def ok_channel(self, token, method, data=None, files=None):
        return {"getMe": {"id": 1, "username": "pub"}, "getChat": {"id": -1, "title": "c"},
                "getChatMember": {"status": "administrator"}}[method]
    from tests.conftest import drain
    monkeypatch.setattr(BotApiAdapter, "_call", ok_channel)
    await client.post(f"/stores/{sid}/channels/connect", headers=h,
                      json={"platform": "telegram", "data": {"bot_token": "222:publish-token", "chat_id": "@c"}})
    await drain()
    monkeypatch.setattr(BotApiAdapter, "_call", fake_bot_call)
    r = await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "222:publish-token"})
    assert r.status_code == 422 and "انتشار" in r.json()["detail"]["message"]

    biz = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "business", "bot_token": "333:biz-token-x"})).json()
    assert biz["status"] == "waiting_business"
    await rt.handle_update(biz["id"], {"update_id": 1, "business_connection": {
        "id": "bc1", "user": {"id": 77, "first_name": "Ali"}, "user_chat_id": 77, "is_enabled": True,
        "rights": {"can_reply": True}, "date": 0}})
    bots = (await client.get(f"/stores/{sid}/chat/bots", headers=h)).json()
    biz = next(b for b in bots if b["mode"] == "business")
    assert biz["status"] == "connected" and biz["can_reply"] and biz["admin_connected"]
    r = await client.post(f"/chat/bots/{biz['id']}/check", headers=h)
    assert r.json()["status"] == "connected"

    # admin link for the standalone bot
    code = (await client.post(f"/chat/bots/{bot['id']}/admin-code", headers=h)).json()
    await rt.handle_update(bot["id"], {"update_id": 2, "message": {
        "message_id": 1, "chat": {"id": 555, "type": "private", "first_name": "Seller"}, "from": {"id": 555},
        "text": f"/admin {code['code']}"}})
    bots = (await client.get(f"/stores/{sid}/chat/bots", headers=h)).json()
    assert next(b for b in bots if b["id"] == bot["id"])["admin_connected"]


async def _customer(bot_id, chat_id, text, msg_id, **extra):
    msg = {"message_id": msg_id, "chat": {"id": chat_id, "type": "private", "first_name": "مریم"},
           "from": {"id": chat_id, "first_name": "مریم"}, "text": text, **extra}
    await rt.handle_update(bot_id, {"update_id": msg_id, "message": msg})


async def test_full_chat_flow(client, monkeypatch):
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550003")
    scheduled: list[int] = []
    monkeypatch.setattr(rt, "schedule", lambda cid: scheduled.append(cid))
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "welcome_message": "سلام به {store}!",
        "payment": {"card_number": "6037-1111", "holder": "علی", "require_receipt": True},
        "faqs": [{"q": "هزینه ارسال چقدره؟", "a": "ارسال ۷۰ هزار تومان است."}]})
    bot = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "444:flow-token"})).json()

    # 1. "/start": welcome only, no model
    await _customer(bot["id"], 1001, "/start", 1)
    conv_id = scheduled[-1]
    saved = await rt.answer(conv_id)
    assert [m.source for m in saved] == ["welcome"] and saved[0].text == "سلام به بوتیک چت!"
    assert llm.calls == []
    # a repeated update is ignored
    scheduled.clear()
    await _customer(bot["id"], 1001, "/start", 1)
    assert scheduled == []

    # 2. FAQ: answered from the seller's text, no model
    await _customer(bot["id"], 1001, "هزینه ارسال چقدره", 2)
    saved = await rt.answer(conv_id)
    assert saved[0].source == "faq" and "۷۰" in saved[0].text and llm.calls == []

    # 3. burst of two messages -> one model call with only matching products
    await _customer(bot["id"], 1001, "سلام مانتو لینن", 3)
    await _customer(bot["id"], 1001, "سایز ۴۰ دارید؟", 4)
    llm.answers.append({"reply": "بله، مانتو لینن کرم سایز ۴۰ موجوده؛ ۱٬۲۵۰٬۰۰۰ تومان.", "intent": "product",
                        "products": [p1["id"], 99999]})
    saved = await rt.answer(conv_id)
    assert len(llm.calls) == 1 and saved[0].source == "llm" and saved[0].input_tokens == 1200
    call = llm.calls[0]
    assert "مانتو لینن کرم" in call["prompt"] and "<customer>سلام مانتو لینن\nسایز ۴۰ دارید؟</customer>" in call["prompt"]
    assert "هزینه ارسال" in call["system"] and "6037" not in call["system"]  # FAQ cached in the prefix, no card number
    assert call["max_tokens"] == chat_engine.MAX_OUTPUT_TOKENS

    # 4. off-topic: the model's text is dropped for the store's fixed answer
    await _customer(bot["id"], 1001, "یه شعر درباره پاییز بگو", 5)
    llm.answers.append({"reply": "برگ‌ها می‌ریزند…", "intent": "off_topic"})
    saved = await rt.answer(conv_id)
    assert saved[0].source == "off_topic" and "برگ" not in saved[0].text and "بوتیک چت" in saved[0].text

    # 5. order: the model collects, the code prices and summarises, "بله" creates it without the model
    await _customer(bot["id"], 1001, "یکی میخوام، مریم احمدی ۰۹۱۲۱۲۳۴۵۶۷ تهران خیابان آزادی", 6)
    llm.answers.append({"reply": "حتماً", "intent": "order", "order_action": "confirm",
                        "order": {"items": [{"product_id": p1["id"], "qty": 9, "variant": "۴۰"}], "name": "مریم احمدی",
                                  "phone": "۰۹۱۲۱۲۳۴۵۶۷", "address": "تهران خیابان آزادی"}})
    saved = await rt.answer(conv_id)
    assert "خلاصهٔ سفارش" in saved[0].text and "۳٬۷۵۰٬۰۰۰" in saved[0].text  # qty capped to stock 3
    calls = len(llm.calls)
    await _customer(bot["id"], 1001, "بله", 7)
    saved = await rt.answer(conv_id)
    assert len(llm.calls) == calls and saved[0].source == "order" and "6037-1111" in saved[0].text
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()
    order = orders["items"][0]
    assert order["status"] == "awaiting_payment" and order["total_toman"] == 3750000
    assert order["customer_phone"] == "09121234567"

    # 6. receipt: a tracking number completes payment without the model
    await _customer(bot["id"], 1001, "کد پیگیری ۱۲۳۴۵۶۷۸", 8)
    saved = await rt.answer(conv_id)
    assert saved[0].source == "receipt" and len(llm.calls) == calls
    order = (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()["items"][0]
    assert order["status"] == "payment_submitted" and order["payment_ref"] == "12345678"
    assert any(m == "sendMessage" and d.get("chat_id") == "1001" for m, d in SENT)

    # seller confirms: stock goes down, customer is told
    r = (await client.patch(f"/ai-orders/{order['id']}", headers=h, json={"status": "confirmed"})).json()
    assert r["status"] == "confirmed"
    assert (await client.get(f"/products/{p1['id']}", headers=h)).json()["stock_qty"] == 0

    # 7. conversation history shows everything, with the order linked
    lst = (await client.get(f"/stores/{sid}/chat/conversations", headers=h)).json()
    assert lst["items"][0]["orders"] == 1 and lst["items"][0]["platform"] == "telegram"
    detail = (await client.get(f"/chat/conversations/{conv_id}", headers=h)).json()
    assert detail["orders"][0]["id"] == order["id"]
    assert {m["source"] for m in detail["messages"] if m["role"] == "ai"} >= {"welcome", "faq", "llm", "off_topic", "order", "receipt"}

    # 8. seller takes over: assistant pauses for that chat
    r = await client.post(f"/chat/conversations/{conv_id}/messages", headers=h, json={"text": "سلام، من فروشنده‌ام"})
    assert r.status_code == 200
    await _customer(bot["id"], 1001, "مرسی از شما، یه سؤال دیگه هم داشتم درباره شال", 9)
    assert await rt.answer(conv_id) == []
    await client.post(f"/chat/conversations/{conv_id}/resume", headers=h)
    llm.answers.append({"reply": "شال نخی مشکی الان ناموجوده.", "intent": "product", "products": [p2["id"]]})
    saved = await rt.answer(conv_id)
    assert saved and saved[0].source == "llm"

    # 9. usage for the seller and the admin
    u = (await client.get(f"/stores/{sid}/chat/usage", headers=h)).json()
    t = u["totals"]
    assert t["llm_calls"] == 4 and t["without_llm"] >= 4 and t["cached_tokens"] == 3200 and t["orders"] == 1
    admin = await login(client, "09120000000")
    au = (await client.get("/admin/chat/usage", headers=admin)).json()
    assert any(s["store_id"] == sid and s["llm_calls"] == 4 for s in au["stores"])
    ov = (await client.get(f"/stores/{sid}/chat/overview", headers=h)).json()
    assert ov["enabled"] and ov["month"]["ai_messages"] == 4


async def test_business_owner_message_pauses_ai(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550004")
    scheduled: list[int] = []
    monkeypatch.setattr(rt, "schedule", lambda cid: scheduled.append(cid))
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    biz = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "business", "bot_token": "555:biz-owner"})).json()
    await rt.handle_update(biz["id"], {"update_id": 1, "business_connection": {
        "id": "bc9", "user": {"id": 77, "first_name": "Ali"}, "user_chat_id": 77, "is_enabled": True,
        "rights": {"can_reply": True}}})
    msg = {"message_id": 10, "business_connection_id": "bc9", "chat": {"id": 3003, "type": "private", "first_name": "سارا"},
           "from": {"id": 3003, "first_name": "سارا"}, "text": "قیمت مانتو چنده؟"}
    await rt.handle_update(biz["id"], {"update_id": 2, "business_message": msg})
    conv_id = scheduled[-1]
    SENT.clear()
    llm.answers.append({"reply": "۱٬۲۵۰٬۰۰۰ تومان", "intent": "product"})
    await rt.answer(conv_id)
    assert all(d.get("business_connection_id") == "bc9" for m, d in SENT if m == "sendMessage")
    # the seller types in their own chat: saved as seller, assistant goes quiet
    own = {**msg, "message_id": 11, "from": {"id": 77, "first_name": "Ali"}, "text": "خودم جواب میدم"}
    scheduled.clear()
    await rt.handle_update(biz["id"], {"update_id": 3, "business_message": own})
    assert scheduled == []
    detail = (await client.get(f"/chat/conversations/{conv_id}", headers=h)).json()
    assert detail["conversation"]["status"] == "human" and detail["messages"][-1]["role"] == "seller"


async def test_test_chat_and_limits(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550005")
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})).json()
    assert [m["source"] for m in r["messages"]] == ["", "welcome"]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "مانتو دارید؟"})).json()
    assert r["messages"][-1]["source"] == "llm" and r["messages"][-1]["input_tokens"] == 1200
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "می‌خوام با ادمین صحبت کنم"})).json()
    assert r["messages"][-1]["source"] == "handoff"
    # tests are not counted in the seller's usage
    u = (await client.get(f"/stores/{sid}/chat/usage", headers=h)).json()
    assert u["totals"]["llm_calls"] == 0
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"reset": True})).json()
    assert r["messages"] == []
    hist = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert hist["messages"] == []

    # monthly plan limit: no model call once it's used up
    monkeypatch.setattr(chat_engine, "chat_limit", lambda db, store: _async(1))
    monkeypatch.setattr(chat_engine, "month_ai_messages", lambda db, sid: _async(1))
    conv = await _real_conv(sid)
    before = len(llm.calls)
    from app.core.db import SessionLocal
    async with SessionLocal() as db:
        c = await db.get(chat_engine.ChatConversation, conv)
        rt.add_message(db, c, role="customer", text="مانتو لینن موجوده؟")
        await db.commit()
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    saved = await rt.answer(conv, send=False)
    assert saved[-1].source == "limit" and len(llm.calls) == before


async def _async(v):
    return v


async def _real_conv(store_id: int) -> int:
    from app.core.db import SessionLocal
    async with SessionLocal() as db:
        c = chat_engine.ChatConversation(store_id=store_id, platform="telegram", mode="bot", external_chat_id="x1",
                                         state={"welcomed": True})
        db.add(c)
        await db.commit()
        return c.id


async def test_generate_chat_logs_cached_tokens(client, monkeypatch):
    admin = await login(client, "09120000000")
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "priced", "kind": "gemini", "api_key": "k-p", "default_model": "gemini-3.6-flash-lite",
        "input_price_per_m": 1_000_000, "output_price_per_m": 4_000_000})).json()
    await client.put("/admin/ai/routes/chat", headers=admin, json={"provider_id": prov["id"], "model": "gemini-3.6-flash-lite"})

    async def fake_call(provider, model, system, prompt, images, temperature, json_mode=True, max_tokens=None, meta=None):
        assert max_tokens == 400
        meta["cached_tokens"] = 1000
        return json.dumps({"reply": "سلام", "intent": "greeting"}), 2000, 100

    monkeypatch.setattr(ai, "_call", fake_call)
    out, usage = await ai.generate_chat(system="s", prompt="p", store_id=None)
    assert out["intent"] == "greeting" and usage["cached_tokens"] == 1000
    # 1000 uncached × 1 + 1000 cached × 0.25 + 100 out × 4 = 1650 per million → 1650 Toman
    assert usage["cost_toman"] == 1650


def test_business_hours_and_faq():
    from datetime import datetime
    hours = {"enabled": True, "days": {"sat": [["09:00", "21:00"]], "fri": [["22:00", "02:00"]]}}
    sat_noon = datetime(2026, 10, 31, 12, 0, tzinfo=chat_engine.TEHRAN)  # a Saturday
    assert chat_engine.is_open(hours, sat_noon)
    assert not chat_engine.is_open(hours, sat_noon.replace(hour=22))
    assert chat_engine.is_open(hours, datetime(2026, 10, 30, 23, 0, tzinfo=chat_engine.TEHRAN))  # Friday night
    assert chat_engine.is_open({"enabled": False}, sat_noon)
    agent = chat_engine.ChatAgent(faqs=[{"q": "هزینه ارسال چقدره؟", "a": "۷۰ هزار"}])
    assert chat_engine.faq_match(agent, "هزينه ارسال چقدره") == "۷۰ هزار"  # Arabic ي normalised
    assert chat_engine.faq_match(agent, "ارسال به شیراز چند روز طول میکشه و هزینه داره؟") == ""


async def test_polling_saves_messages_and_moves_offset(client, monkeypatch):
    import asyncio
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550006")
    scheduled: list[int] = []
    monkeypatch.setattr(rt, "schedule", lambda cid: scheduled.append(cid))
    bot = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "bale", "mode": "bot", "bot_token": "666:poll-token"})).json()
    stop = asyncio.Event()
    seen: list[int] = []

    async def fake_updates(b, offset, timeout=25):
        seen.append(offset)
        if len(seen) > 1:
            stop.set()
            return []
        return [{"update_id": 40, "message": {"message_id": 1, "chat": {"id": 7, "type": "private", "first_name": "رضا"},
                                              "from": {"id": 7}, "text": "سلام"}},
                {"update_id": 41, "message": {"message_id": 2, "chat": {"id": -5, "type": "group"}, "text": "spam"}}]

    monkeypatch.setattr(rt.tg, "get_updates", fake_updates)
    await asyncio.wait_for(rt._poll_bot(bot["id"], stop), 5)
    assert seen == [0, 42] and len(scheduled) == 1  # the group message is ignored
    lst = (await client.get(f"/stores/{sid}/chat/conversations?platform=bale", headers=h)).json()
    assert lst["items"][0]["customer_name"] == "رضا"


async def test_catalog_reaches_the_model(client, monkeypatch):
    """Draft (imported) and out-of-stock products are still the seller's catalog; suffixed words still match."""
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550009")
    from app.core.db import SessionLocal
    from app.models import Product
    async with SessionLocal() as db:
        db.add(Product(store_id=sid, title="کفش اسپرت سفید", price=900000, stock_qty=0, status="draft"))
        await db.commit()
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "چه محصولاتی دارید؟"})
    prompt = llm.calls[-1]["prompt"]
    assert "مانتو لینن کرم" in prompt and "کفش اسپرت سفید" in prompt
    assert "Never say you have no access" in llm.calls[-1]["system"]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "کفشاتون چند؟"})
    assert "کفش اسپرت سفید" in llm.calls[-1]["prompt"].split("PRODUCTS")[1].splitlines()[1]


async def test_faq_extract_from_text_and_file(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550010")
    monkeypatch.setattr("app.services.chat_faq.ai.generate_chat", llm)
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"faqs": [{"q": "هزینه ارسال چقدر است؟", "a": "۵۰ هزار"}]})
    llm.answers = [{"items": [{"q": "هزینه ارسال چقدر است؟", "a": "تکراری"},
                              {"q": "چند روزه می‌رسد؟", "a": "۲ تا ۴ روز کاری"}, {"q": "", "a": "x"}]}]
    r = await client.post(f"/stores/{sid}/chat/faq/extract", headers=h,
                          data={"text": "ارسال به تهران ۲ تا ۴ روز کاری طول می‌کشد و هزینه ارسال ۵۰ هزار تومان است."})
    assert r.status_code == 200, r.text
    assert r.json()["items"] == [{"q": "چند روزه می‌رسد؟", "a": "۲ تا ۴ روز کاری"}]
    assert "<doc>" in llm.calls[-1]["prompt"] and "ارسال به تهران" in llm.calls[-1]["prompt"]

    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:t>مرجوعی تا ۷ روز امکان‌پذیر است</w:t></w:p></w:document>")
    llm.answers = [{"items": [{"q": "مرجوعی دارید؟", "a": "بله، تا ۷ روز"}]}]
    r = await client.post(f"/stores/{sid}/chat/faq/extract", headers=h,
                          files={"file": ("faq.docx", buf.getvalue(), "application/octet-stream")})
    assert r.json()["items"][0]["q"] == "مرجوعی دارید؟" and "مرجوعی تا ۷ روز" in llm.calls[-1]["prompt"]
    r = await client.post(f"/stores/{sid}/chat/faq/extract", headers=h, files={"file": ("a.xlsx", b"PK", "application/octet-stream")})
    assert r.status_code == 422


async def test_assistant_keeps_the_chat_unless_a_person_is_wanted(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550011")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    conv = await _real_conv(sid)
    from app.core.db import SessionLocal

    async def say(text):
        async with SessionLocal() as db:
            rt.add_message(db, await db.get(chat_engine.ChatConversation, conv), role="customer", text=text)
            await db.commit()
        saved = await rt.answer(conv, send=False)
        async with SessionLocal() as db:
            return saved, (await db.get(chat_engine.ChatConversation, conv)).status

    # an unsure model answer is not a handoff: the reply is sent and the assistant stays on
    llm.answers = [{"reply": "این رو دقیق نمی‌دونم، ولی مانتو لینن موجوده.", "intent": "product", "handoff": True}]
    saved, status = await say("جنس آستر مانتو چیه؟")
    assert saved[-1].source == "llm" and status == "ai"
    llm.answers = [{"reply": "", "intent": "faq", "handoff": True}]
    saved, status = await say("پشتیبانی دارید؟")
    assert saved[-1].text == chat_engine.NO_ANSWER_TEXT and status == "ai"
    # a real request for a person pauses the assistant (no model call)
    before = len(llm.calls)
    saved, status = await say("وصلم کن به پشتیبانی")
    assert saved[-1].source == "handoff" and status == "needs_human" and len(llm.calls) == before


async def test_photos_variants_and_clarify(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550012")
    from app.core.db import SessionLocal
    from app.models import Product, ProductImage
    async with SessionLocal() as db:
        hoodie = Product(store_id=sid, title="هودی کلاه‌دار", price=890000, stock_qty=4,
                         attributes={"جنس": "دورس", "برند": "X", "کشور": "ایران", "یقه": "کلاه‌دار",
                                     "آستین": "بلند", "رنگ": "مشکی، نارنجی، طوسی", "سایز": "L، XL"})
        db.add(hoodie)
        await db.flush()
        db.add_all([ProductImage(product_id=hoodie.id, url=f"/media/x/h{i}.jpg", position=i) for i in range(4)])
        await db.commit()
        hid = hoodie.id
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "این هودی کلاه‌دار ماست 👇", "intent": "product", "products": [hid], "images": [hid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکس هودی رو بفرست"})).json()
    call = llm.calls[-1]
    # colour/size come first in the product line even with many details, and the photo count is shown
    line = next(x for x in call["prompt"].splitlines() if x.startswith(f"#{hid} "))
    assert "رنگ: مشکی، نارنجی، طوسی" in line and "سایز: L، XL" in line and "۴ عکس" in line
    assert "CLARIFY" in call["system"] and "VARIANTS" in call["system"]
    imgs = [m for m in r["messages"] if m["image_url"]]
    assert len(imgs) == 1 and imgs[0]["image_url"] == "/media/x/h0.jpg" and "هودی کلاه‌دار" in imgs[0]["text"]
    llm.answers = [{"reply": "بفرما", "intent": "product", "products": [hid], "images": [hid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکس‌های بیشتر"})).json()
    assert len([m for m in r["messages"][-4:] if m["image_url"]]) == 3
    assert f"LAST PRODUCT: #{hid}" in llm.calls[-1]["prompt"]
    # an id the model made up gets no photo
    llm.answers = [{"reply": "بفرما", "intent": "product", "images": [99999]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکسشو ببینم"})).json()
    assert not r["messages"][-1]["image_url"]


async def test_answer_rules_and_stock_only_available(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550013")
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"answer_policy": {
        "stock": "available", "rules": ["  تخفیف نده  ", "", "به رقبا اشاره نکن"]}})
    assert r.json()["answer_policy"] == {"stock": "available", "rules": ["تخفیف نده", "به رقبا اشاره نکن"]}
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "بله، فقط ۳ تا ازش مونده 🌸", "intent": "product"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "مانتو لینن موجوده؟"})).json()
    call = llm.calls[-1]
    assert "1. تخفیف نده" in call["system"] and "2. به رقبا اشاره نکن" in call["system"]
    assert "مانتو لینن کرم" in call["prompt"] and "موجودی" not in call["prompt"].split("RECENT CHAT")[0] and "| موجود |" in call["prompt"]
    assert r["messages"][-1]["text"] == "بله، موجوده 🌸"
    # default: exact numbers
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"answer_policy": {"stock": "exact", "rules": []}})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "مانتو لینن موجوده؟"})
    assert "موجودی 3" in llm.calls[-1]["prompt"] and "SELLER RULES" not in llm.calls[-1]["system"]


async def test_generate_chat_survives_prose_and_cut_json(client, monkeypatch):
    admin = await login(client, "09120000000")
    prov = (await client.post("/admin/ai/providers", headers=admin, json={
        "name": "or", "kind": "openrouter", "api_key": "k-o", "default_model": "deepseek/x"})).json()
    await client.put("/admin/ai/routes/chat", headers=admin, json={"provider_id": prov["id"], "model": "deepseek/x",
                                                                  "temperature": 90})
    outputs = ["oops not json", 'Here you go: {"reply":"بله موجوده","intent":"product","products":[1', ]
    temps = []

    async def fake_call(provider, model, system, prompt, images, temperature, json_mode=True, max_tokens=None, meta=None):
        temps.append(temperature)
        return outputs.pop(0), 100, 10

    monkeypatch.setattr(ai, "_call", fake_call)
    out, usage = await ai.generate_chat(system="s", prompt="p", store_id=None)
    assert out["reply"] == "بله موجوده" and usage["input_tokens"] == 200
    assert temps[0] <= ai.CHAT_MAX_TEMPERATURE and temps[1] == 0.0


async def test_real_conversation_from_the_screenshots(client, monkeypatch):
    """ali's test: forwarded hoodie post, "موجوده؟", sizes, "similar?" — with a model that fails once and mixes languages."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550014")
    from app.core.db import SessionLocal
    from app.models import Product
    async with SessionLocal() as db:
        hoodie = Product(store_id=sid, title="هودی کلاهدار قهوه‌ای با درزهای عمودی", price=100000, stock_qty=2,
                         category_hint="هودی", attributes={"رنگ": "قهوه‌ای", "برش": "گشاد"})
        other = Product(store_id=sid, title="هودی زیپ‌دار مشکی", price=980000, stock_qty=1, category_hint="هودی")
        db.add_all([hoodie, other])
        await db.commit()
        hid, oid = hoodie.id, other.id
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    # the forwarded post names the hoodie: it becomes LAST PRODUCT even before the model says so
    llm.answers = [{"reply": "این هودی قهوه‌ای کلاهدار ماست 🌸", "intent": "product"}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={
        "text": "هودی کلاهدار به رنگ قهوه‌ای شکلاتی با برش گشاد و آستین بلند. درزهای عمودی"})

    # the model fails once: a true answer about that hoodie, not "a colleague will answer"
    async def boom(**kw):
        raise ai.AIError("پاسخ هوش مصنوعی قابل خواندن نبود.", code="ai_bad_json")
    monkeypatch.setattr(chat_engine.ai, "generate_chat", boom)
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "موجوده؟"})).json()
    text = r["messages"][-1]["text"]
    assert "هودی کلاهدار قهوه‌ای" in text and "موجوده" in text and "۱۰۰٬۰۰۰" in text and "همکارم" not in text
    monkeypatch.setattr(chat_engine.ai, "generate_chat", llm)

    # mixed-language answer: asked again once, then cleaned
    llm.answers = [{"reply": "unfortunately سایزبندی ثبت نشده確認", "intent": "product"},
                   {"reply": "سایزبندی ثبت نشده، ولی برشش گشاده 🌸 unfortunately", "intent": "product"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سایزبندیش چیه؟"})).json()
    assert r["messages"][-1]["text"] == "سایزبندی ثبت نشده، ولی برشش گشاده 🌸"
    assert "Persian only" in llm.calls[-1]["prompt"]

    # "similar?" brings the other hoodie into PRODUCTS
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "چی دیگه دارین مشابه؟"})
    prompt = llm.calls[-1]["prompt"]
    assert f"#{oid} " in prompt and f"LAST PRODUCT: #{hid}" in prompt


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


async def test_gemini3_chat_drops_rejected_thinking_config_and_retries_when_cut(monkeypatch):
    """ali's Metis route (gemini-3.5-flash-lite) answered a bare 400 "invalid argument" to the thinking setting."""
    from app.models import AiProvider
    from app.core.security import encrypt_credentials
    prov = AiProvider(name="m", kind="gemini", api_key_encrypted=encrypt_credentials({"api_key": "k"}), base_url="")
    bodies = []
    replies = [
        _Resp(400, {"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}}),
        _Resp(200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {}}),
        _Resp(200, {"candidates": [{"content": {"parts": [{"text": '{"reply":"بله موجوده"}'}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}),
    ]

    async def fake_post(self, url, headers=None, json=None):
        bodies.append(__import__("copy").deepcopy(json))
        return replies.pop(0)

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    text, tin, tout = await ai._call(prov, "gemini-3.5-flash-lite", "s", "p", [], 0.3, max_tokens=700)
    assert text == '{"reply":"بله موجوده"}'
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "thinkingConfig" not in bodies[1]["generationConfig"]
    assert bodies[2]["generationConfig"]["maxOutputTokens"] == 4096


async def test_openai_compatible_retries_an_answer_lost_to_reasoning(monkeypatch):
    from app.models import AiProvider
    from app.core.security import encrypt_credentials
    prov = AiProvider(name="d", kind="openai", api_key_encrypted=encrypt_credentials({"api_key": "k"}),
                      base_url="https://x.test/v1")
    caps = []
    replies = [_Resp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}),
               _Resp(200, {"choices": [{"message": {"content": '{"reply":"سلام"}'}, "finish_reason": "stop"}], "usage": {}})]

    async def fake_post(self, url, headers=None, json=None):
        caps.append(json["max_tokens"])
        return replies.pop(0)

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    text, *_ = await ai._call(prov, "deepseek-flash", "s", "p", [], 0.3, max_tokens=700)
    assert text == '{"reply":"سلام"}' and caps == [700, 4096]
