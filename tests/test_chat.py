"""Sales chat assistant: connections, the cheap no-model paths, one model call per turn, orders, takeover, usage."""

import re
import json

import pytest
from sqlalchemy import delete, select

from app.services.chat_router import route as REAL_ROUTE  # _setup stubs the router; its own test uses the real one

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
    from app.services import chat_router

    async def no_route(*a, **kw):
        return ""  # the router's own tests set its answers
    monkeypatch.setattr(chat_router, "route", no_route)
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

    # 2. FAQ: the model reads the question first and answers from the seller's FAQ
    await _customer(bot["id"], 1001, "هزینه ارسال چقدره", 2)
    llm.answers.append({"reply": "ارسال ۷۰ هزار تومانه 🌸", "intent": "faq"})
    saved = await rt.answer(conv_id)
    assert saved[0].source == "llm" and "۷۰" in saved[0].text and len(llm.calls) == 1
    assert "هزینه ارسال" in llm.calls[0]["system"]

    # 3. burst of two messages -> one model call with only matching products
    await _customer(bot["id"], 1001, "سلام مانتو لینن", 3)
    await _customer(bot["id"], 1001, "سایز ۴۰ دارید؟", 4)
    llm.answers.append({"reply": "بله، مانتو لینن کرم سایز ۴۰ موجوده؛ ۱٬۲۵۰٬۰۰۰ تومان.", "intent": "product",
                        "products": [p1["id"], 99999]})
    saved = await rt.answer(conv_id)
    assert len(llm.calls) == 2 and saved[0].source == "llm" and saved[0].input_tokens == 1200
    call = llm.calls[1]
    assert "مانتو لینن کرم" in call["prompt"] and "<customer>سلام مانتو لینن\nسایز ۴۰ دارید؟</customer>" in call["prompt"]
    assert "هزینه ارسال" in call["system"] and "6037" not in call["system"]  # FAQ cached in the prefix, no card number
    assert call["max_tokens"] == chat_engine.MAX_OUTPUT_TOKENS

    # 4. off-topic: the model's text is dropped for the store's fixed answer
    await _customer(bot["id"], 1001, "یه شعر درباره پاییز بگو", 5)
    llm.answers.append({"reply": "برگ‌ها می‌ریزند…", "intent": "off_topic"})
    saved = await rt.answer(conv_id)
    assert saved[0].source == "off_topic" and "برگ" not in saved[0].text and "بوتیک چت" in saved[0].text

    # 5. order: the model collects, the code prices and summarises; the model reads "بله" and the code places it
    await _customer(bot["id"], 1001, "یکی میخوام، مریم احمدی ۰۹۱۲۱۲۳۴۵۶۷ تهران خیابان آزادی", 6)
    llm.answers.append({"reply": "حتماً", "intent": "order", "order_action": "confirm",
                        "order": {"items": [{"product_id": p1["id"], "qty": 9, "variant": "۴۰"}], "name": "مریم احمدی",
                                  "phone": "۰۹۱۲۱۲۳۴۵۶۷", "address": "تهران خیابان آزادی"}})
    saved = await rt.answer(conv_id)
    assert "خلاصهٔ سفارش" in saved[0].text and "۳٬۷۵۰٬۰۰۰" in saved[0].text  # qty capped to stock 3
    await _customer(bot["id"], 1001, "بله", 7)
    llm.answers.append({"reply": "", "intent": "order", "order_action": "confirm"})
    saved = await rt.answer(conv_id)
    assert "STEP: summary" in llm.calls[-1]["prompt"]
    assert saved[0].source == "order" and "6037-1111" in saved[0].text
    calls = len(llm.calls)
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
    assert {m["source"] for m in detail["messages"] if m["role"] == "ai"} >= {"welcome", "llm", "off_topic", "order", "receipt"}

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
    assert t["llm_calls"] == 6 and t["without_llm"] >= 2 and t["cached_tokens"] == 4800 and t["orders"] == 1
    admin = await login(client, "09120000000")
    au = (await client.get("/admin/chat/usage", headers=admin)).json()
    assert any(s["store_id"] == sid and s["llm_calls"] == 6 for s in au["stores"])
    ov = (await client.get(f"/stores/{sid}/chat/overview", headers=h)).json()
    assert ov["enabled"] and ov["month"]["ai_messages"] == 6


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
    llm.answers.append({"reply": "", "intent": "handoff", "handoff": True, "handoff_reason": "asked_human"})
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


def test_business_hours():
    from datetime import datetime
    hours = {"enabled": True, "days": {"sat": [["09:00", "21:00"]], "fri": [["22:00", "02:00"]]}}
    sat_noon = datetime(2026, 10, 31, 12, 0, tzinfo=chat_engine.TEHRAN)  # a Saturday
    assert chat_engine.is_open(hours, sat_noon)
    assert not chat_engine.is_open(hours, sat_noon.replace(hour=22))
    assert chat_engine.is_open(hours, datetime(2026, 10, 30, 23, 0, tzinfo=chat_engine.TEHRAN))  # Friday night
    assert chat_engine.is_open({"enabled": False}, sat_noon)


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
    n = len(llm.calls)
    llm.answers.append({"reply": "", "intent": "catalog"})
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "چه محصولاتی دارید؟"})).json()
    # the model reads "what do you have?" and the catalog itself gives the list
    assert len(llm.calls) == n + 1 and r["messages"][-1]["source"] == "catalog"
    assert "مانتو لینن کرم" in r["messages"][-1]["text"] and "کفش اسپرت سفید" in r["messages"][-1]["text"]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "یه چیز خوب پیشنهاد بده"})
    prompt = llm.calls[-1]["prompt"]
    assert "مانتو لینن کرم" in prompt and "کفش اسپرت سفید" in prompt
    assert "Never say you can't see products" in llm.calls[-1]["system"]
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
    # a real request for a person (the model reads it as one) pauses the assistant
    llm.answers = [{"reply": "", "intent": "handoff", "handoff": True, "handoff_reason": "asked_human"}]
    saved, status = await say("وصلم کن به پشتیبانی")
    assert saved[-1].source == "handoff" and status == "needs_human"


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
    assert "WHICH PRODUCT" in call["system"] and "VARIANTS" in call["system"]
    imgs = [m for m in r["messages"] if m["image_url"]]
    assert len(imgs) == 1 and imgs[0]["image_url"] == "/media/x/h0.jpg" and "هودی کلاه‌دار" in imgs[0]["text"]
    llm.answers = [{"reply": "بفرما", "intent": "product", "products": [hid], "images": [hid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکس‌های بیشتر"})).json()
    assert len([m for m in r["messages"][-4:] if m["image_url"]]) == 3
    assert f"LAST PRODUCT: #{hid}" in llm.calls[-1]["prompt"]
    # the model offers a photo but the customer did not ask: no photo yet
    llm.answers = [{"reply": "بله موجوده، عکسش رو بفرستم؟", "intent": "product", "images": [hid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "موجوده؟"})).json()
    assert not any(m["image_url"] for m in r["messages"][-2:])
    # an id the model made up gets no photo
    llm.answers = [{"reply": "بفرما", "intent": "product", "images": [99999]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکسشو ببینم"})).json()
    assert not any(m["image_url"] for m in r["messages"][-2:])


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
    assert "موجودی 3" in llm.calls[-1]["prompt"] and "SELLER RULES (obey" not in llm.calls[-1]["system"]


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


async def test_photo_of_the_named_product_and_yes_to_a_photo_offer(client, monkeypatch):
    """ali's run: "photos of the brown hoodie" sent the black octopus one; "ارهه" to "send its photo?" got nothing."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550015")
    from app.core.db import SessionLocal
    from app.models import Product, ProductImage
    async with SessionLocal() as db:
        brown = Product(store_id=sid, title="هودی کلاهدار قهوه‌ای", price=100000, stock_qty=2)
        octo = Product(store_id=sid, title="هودی اختاپوس", price=0, stock_qty=1)
        sully = Product(store_id=sid, title="هودی سالیوان", price=890000, stock_qty=3)
        db.add_all([brown, octo, sully])
        await db.flush()
        db.add_all([ProductImage(product_id=p.id, url=f"/media/x/{p.id}.jpg") for p in (brown, octo, sully)])
        await db.commit()
        b, o, s = brown.id, octo.id, sully.id
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "عالیه 🌸 عکس‌های هودی قهوه‌ای رو ببینید.", "intent": "product", "images": [o]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکس هودی قهوه‌ای رو بفرست"})).json()
    imgs = [m for m in r["messages"] if m["image_url"]]
    assert [m["image_url"] for m in imgs] == [f"/media/x/{b}.jpg"] and imgs[0]["text"] == "هودی کلاهدار قهوه‌ای — ۱۰۰٬۰۰۰ تومان"

    llm.answers = [{"reply": "بله هودی سالیوان رو داریم 🌸 عکسش رو بفرستم؟", "intent": "product", "products": [s]}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "هودی با طرح سالیوان؟"})
    # "ارهه" to the offer: the model reads it as yes and names the photo to send
    llm.answers = [{"reply": "بفرمایید 🌸", "intent": "product", "images": [s]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "ارهه"})).json()
    assert r["messages"][-2]["image_url"] == f"/media/x/{s}.jpg" and r["messages"][-2]["text"] == "هودی سالیوان — ۸۹۰٬۰۰۰ تومان"
    # a photo of a product without a price has no "unknown price" caption
    llm.answers = [{"reply": "بفرما", "intent": "product", "images": [o]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عکس هودی اختاپوس"})).json()
    assert r["messages"][-2]["text"] == "هودی اختاپوس" and r["messages"][-1]["text"] == "بفرما"  # photo first, then words


async def test_busy_buyer_is_not_handed_over(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550016")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    conv = await _real_conv(sid)
    from app.core.db import SessionLocal
    for i in range(12):  # the old limit (10 in 10 minutes) handed this chat to the admin
        async with SessionLocal() as db:
            rt.add_message(db, await db.get(chat_engine.ChatConversation, conv), role="customer", text=f"مانتو لینن سایز {i}؟")
            await db.commit()
        saved = await rt.answer(conv, send=False)
        assert saved[-1].source == "llm"
    async with SessionLocal() as db:
        assert (await db.get(chat_engine.ChatConversation, conv)).status == "ai"


async def test_size_chart_offer_waits_for_yes(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550017")
    from app.core.db import SessionLocal
    from app.models import ChatReference
    async with SessionLocal() as db:
        ref = ChatReference(store_id=sid, title="راهنمای سایز", text="جدول سایز: M=۱۰۰ L=۱۰۴")
        db.add(ref)
        await db.commit()
        rid = ref.id
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "برشش گشاده 🌸 راهنمای سایز رو براتون بفرستم؟", "intent": "product", "refs": [rid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سایز مانتو لینن چطوره؟"})).json()
    assert "جدول سایز" not in r["messages"][-1]["text"]
    llm.answers = [{"reply": "بفرمایید 🌸", "intent": "faq", "refs": [rid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "آره لطفا"})).json()
    assert r["messages"][-1]["text"] == "جدول سایز: M=۱۰۰ L=۱۰۴"
    # asked directly: the chart comes with the answer
    llm.answers = [{"reply": "اینم جدول سایز 🌸", "intent": "faq", "refs": [rid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "جدول سایزتون رو بفرست"})).json()
    assert r["messages"][-1]["text"] == "جدول سایز: M=۱۰۰ L=۱۰۴"


async def test_swapped_product_keeps_size_and_ok_is_not_a_photo_request(client, monkeypatch):
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550018")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    from app.core.db import SessionLocal
    from app.models import ProductImage
    async with SessionLocal() as db:
        db.add(ProductImage(product_id=p1["id"], url="/media/x/m.jpg"))
        await db.commit()
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام وقت بخیر"})
    assert llm.calls == []  # a longer greeting is still a greeting
    llm.answers = [{"reply": "ثبت شد", "intent": "order", "products": [p1["id"]],
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "XL"}]}, "order_action": "update"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "اوکی مانتو لینن سایز XL"})).json()
    assert not any(m["image_url"] for m in r["messages"][-2:])
    llm.answers = [{"reply": "عوض شد", "intent": "order",
                    "order": {"items": [{"product_id": p2["id"], "qty": 1}]}, "order_action": "update"}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "نه بجاش شال نخی"})
    async with SessionLocal() as db:
        from sqlalchemy import select
        conv = (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first()
        assert conv.state["order"]["items"] == [{"product_id": p2["id"], "qty": 1, "variant": "XL"}]


async def _order_ready(client, h, sid, llm, p1, extra_items=None):
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"reset": True})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    items = [{"product_id": p1["id"], "qty": 1, "variant": "L"}] + (extra_items or [])
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": {
        "items": items, "name": "سارا کریمی", "phone": "09171234567", "address": "شیراز، معالی آباد"}}]
    return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "همین رو ثبت کن"})).json()


async def test_order_yes_variants_payment_ref_and_duplicates(client, monkeypatch):
    """From two long buyer chats: "بله تایید"/"تایید" re-showed the summary, a payment reference sent before
    the yes was lost, and the model listed one item twice."""
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550019")
    r = await _order_ready(client, h, sid, llm, p1, [{"product_id": p1["id"], "qty": 1, "variant": "L"}])
    assert "× ۱ " in r["messages"][-1]["text"] and r["messages"][-1]["text"].count("مانتو لینن کرم") == 1
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "بله تایید"})).json()
    assert r["messages"][-1]["source"] == "order"

    await _order_ready(client, h, sid, llm, p1)
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "کارت به کارت کردم، شماره پیگیری ۵۵۱۲۰۹۸۸"})).json()
    assert r["messages"][-1]["source"] == "receipt"
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert any(o.get("payment_ref") == "55120988" and o["status"] == "payment_submitted" for o in rows)


async def test_declined_chart_is_not_sent_later(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550020")
    from app.core.db import SessionLocal
    from app.models import ChatReference
    async with SessionLocal() as db:
        ref = ChatReference(store_id=sid, title="جدول سایز بندی", text="جدول: M L XL")
        db.add(ref)
        await db.commit()
        rid = ref.id
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "برای شما M مناسبه. جدول سایز رو بفرستم؟", "intent": "product", "refs": [rid]}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "چه سایزی بگیرم؟"})
    llm.answers = [{"reply": "باشه، M براتون مناسبه.", "intent": "product"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "نه جدول نمیخوام، خودت بگو"})).json()
    assert "جدول: M L XL" not in [m["text"] for m in r["messages"]]
    llm.answers = [{"reply": "بفرمایید", "intent": "faq", "refs": [rid]}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "راستی جدول سایز رو بفرست"})).json()
    assert r["messages"][-1]["text"] == "جدول: M L XL"


async def test_custom_print_shop_collects_the_design_and_codes(client, monkeypatch):
    """A hoodie shop that prints the customer's own design: the picture goes on the order; product codes work."""
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550021")
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"},
        "order_fields": [{"label": "طرح چاپ", "kind": "image", "required": True, "hint": "عکس یا فایل طرح"},
                         {"label": "متن روی لباس", "kind": "text", "required": False}, {"label": ""}]})
    assert [f["label"] for f in r.json()["order_fields"]] == ["طرح چاپ", "متن روی لباس"]
    await client.patch(f"/products/{p1['id']}", headers=h, json={"sku": "H12"})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    llm.answers = [{"reply": "مانتو لینن کرم با کد H12 🌸", "intent": "product", "products": [p1["id"]]}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "کد h12 رو میخوام"})
    call = llm.calls[-1]
    assert "EXTRA ORDER FIELDS" in call["system"] and "طرح چاپ (image, required)" in call["system"]
    assert f"#{p1['id']} | مانتو لینن کرم (کد H12)" in call["prompt"]
    # all usual fields known but no design yet: no confirm
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": {
        "items": [{"product_id": p1["id"], "qty": 1, "variant": "L"}], "name": "نیما", "phone": "09120000001",
        "address": "مشهد", "extras": {"متن روی لباس": "NIMA", "ناشناخته": "x"}}}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "ثبتش کن"})).json()
    assert "خلاصهٔ سفارشت" not in r["messages"][-1]["text"]
    # the customer sends the design picture: it lands on the order and the model is told so
    from app.core.db import SessionLocal
    from sqlalchemy import select
    async with SessionLocal() as db:
        conv = (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first()
        rt.add_message(db, conv, role="customer", text="", image_url="/media/x/design.png")
        await db.commit()
        cid = conv.id
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    saved = await rt.answer(cid, send=False)
    assert "طرح چاپ" in llm.calls[-1]["prompt"] and "خلاصهٔ سفارشت" in saved[-1].text
    assert "📝 طرح چاپ: ✔️ عکس رسید" in saved[-1].text and "📝 متن روی لباس: NIMA" in saved[-1].text
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "بله"})).json()
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert rows[0]["extras"] == {"متن روی لباس": "NIMA", "طرح چاپ": "/media/x/design.png"}


async def test_every_question_is_read_by_the_model_first(client, monkeypatch):
    """The usual test chat: the catalog question, shipping, off-topic, shoes: each one goes to the model, which decides."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550031")

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "", "intent": "catalog"}]
    m = await say("چه محصولاتی دارید؟")
    assert m["source"] == "catalog" and "مانتو لینن کرم" in m["text"] and "(ناموجود)" in m["text"]
    assert len(llm.calls) == 1 and '"catalog"' in llm.calls[-1]["system"]

    llm.answers = [{"reply": "هزینه ارسال بعد از ثبت سفارش اعلام می‌شه.", "intent": "faq"}] * 2
    assert (await say("هزینه ارسال چقدره؟"))["source"] == "llm"
    assert (await say("هزینه ارسال چقدره"))["source"] == "llm"
    assert len(llm.calls) == 3

    llm.answers = [{"reply": "برگ‌ها…", "intent": "off_topic"}]
    m = await say("یه شعر برام بگو")
    assert m["source"] == "off_topic" and "برگ" not in m["text"]

    llm.answers = [{"reply": "کفش نداریم 🙏 مانتو لینن کرم داریم.", "intent": "product"}]
    m = await say("کفش هم دارین؟")
    assert "نداریم" in m["text"] and m["source"] == "llm"
    system = llm.calls[-1]["system"]
    assert "NOT off_topic" in system and "shipping costs" in system and len(llm.calls) == 5


async def test_orders_off_hands_the_buyer_to_the_admin(client, monkeypatch):
    """AI ordering off: "shall I send it?" + "yes" passes the chat to the admin with the choice, no loop."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550032")

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    system_seen = []
    llm.answers = [{"reply": "عالیه، سایز XL هودی طوسی رو براتون ارسال کنم؟", "intent": "product", "products": [p1["id"]],
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "XL"}]}, "order_action": "update"}]
    await say("مانتو لینن کرم XL")
    system_seen.append(llm.calls[-1]["system"])
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    m = await say("اره")
    assert m["source"] == "handoff" and "همکارم" in m["text"]
    assert "never take or register orders" in system_seen[0]
    convs = (await client.get(f"/stores/{sid}/chat/conversations", headers=h)).json()
    assert convs  # the conversation exists; the admin note carries the choice
    from app.core.db import SessionLocal
    from app.models import Notification
    from sqlalchemy import select
    async with SessionLocal() as db:
        notes = [x.title_fa + x.body_fa for x in (await db.execute(select(Notification).where(
            Notification.store_id == sid))).scalars()]
    assert any("مانتو لینن کرم" in t and "XL" in t for t in notes), notes

    # the model says the customer wants to buy: same hand-over
    llm.answers = [{"reply": "", "intent": "order", "order": {"items": [{"product_id": p1["id"], "qty": 1}]},
                    "order_action": "confirm"}]
    m = await say("همینو می‌خوام بخرم")
    assert m["source"] == "handoff" and m["intent"] == "order"

    llm.answers = [{"reply": "", "intent": "handoff", "handoff": True, "handoff_reason": "asked_human"}]
    assert (await say("ادمین اصلی کجاس"))["source"] == "handoff"
    llm.answers = [{"reply": "باشه 🌸", "intent": "product"}]
    assert (await say("باشه"))["source"] == "llm"  # "ok" is not "thanks"


async def test_adding_to_a_paid_order_asks_only_for_the_rest(client, monkeypatch):
    """ali's chat: ordered, sent the receipt, then "add another hoodie" — the bot must ask for the new amount only."""
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550033")

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await _order_ready(client, h, sid, llm, p1)
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    assert (await say("بله"))["source"] == "order"
    assert (await say("56455561561561"))["source"] == "receipt"

    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm",
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "M"}]}}]
    m = await say("همون مدل سایز M رو هم اضافه کن")
    assert "PLACED ORDER #" in llm.calls[-1]["prompt"] and "paid" in llm.calls[-1]["prompt"]
    assert "اضافه به سفارش" in m["text"] and "۱٬۲۵۰٬۰۰۰" in m["text"] and "سارا" not in m["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    m = await say("بله")
    assert "اضافه شد" in m["text"] and "۱٬۲۵۰٬۰۰۰" in m["text"] and "2,500" not in m["text"]

    # "thanks" while the addition is unpaid: the model sees what is still owed
    llm.answers = [{"reply": "خواهش می‌کنم 🌸 فقط مبلغ ۱٬۲۵۰٬۰۰۰ تومان رو پرداخت کن.", "intent": "greeting"}]
    m = await say("مرسی")
    assert "unpaid" in llm.calls[-1]["prompt"] and "STEP: order #" in llm.calls[-1]["prompt"]

    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    o = rows[0]
    assert len(o["items"]) == 2 and [i["paid"] for i in o["items"]] == [True, False]
    assert o["total_toman"] == 2500000 and o["due_toman"] == 1250000 and o["status"] == "awaiting_payment"

    assert (await say("کد پیگیری ۷۷۸۸۹۹۰۰"))["source"] == "receipt"
    rows = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    assert rows[0]["due_toman"] == 0 and all(i["paid"] for i in rows[0]["items"])


async def test_order_form_and_address_is_not_off_topic(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550034")
    form = "برای ثبت سفارش این‌ها رو بفرست 🌸\nنام:\nموبایل:\nآدرس:"
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890", "order_form": form}})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "عالیه، لطفاً نام و شماره تماستون رو بفرمایید.", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "XL"}]}}]
    assert (await say("اره همینو میخوام"))["text"] == form
    # the model reads the filled-in form; with every detail known the code shows the summary
    llm.answers = [{"reply": "ممنون 🌸", "intent": "order", "order_action": "update", "order": {
        "name": "علی رضایی", "phone": "۰۹۱۲۳۳۳۳۴۳۴", "address": "قم خیابان صفاییه کوچه ۱۲"}}]
    m = await say("نام: علی رضایی\nموبایل: ۰۹۱۲۳۳۳۳۴۳۴\nآدرس: قم خیابان صفاییه کوچه ۱۲")
    assert "خلاصهٔ سفارشت" in m["text"]
    assert "علی رضایی" in m["text"] and "09123333434" in m["text"] and "قم خیابان صفاییه" in m["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    assert (await say("بله"))["source"] == "order"


async def test_contact_details_one_by_one_are_read_by_the_model(client, monkeypatch):
    """ali's chat: "خب بفرستین برام" was taken as the name and "." as the address. Now the model reads each message;
    only what it takes as a name/phone/address goes on the order, and the summary comes when all are there."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550035")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm",
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "۴۰"}]}}]
    m = await say("اره همینو میخوام")
    assert "نام و نام خانوادگی" in m["text"] and "آدرس کامل" in m["text"]
    llm.answers = [{"reply": "حتماً 🌸 فقط اسم، شماره و آدرست رو بفرست.", "intent": "order"}]
    m = await say("خب بفرستین برام")
    assert m["text"] == "حتماً 🌸 فقط اسم، شماره و آدرست رو بفرست."
    llm.answers = [{"reply": "ممنون، اسم و آدرس؟", "intent": "order", "order_action": "update",
                    "order": {"phone": "09123333434"}}]
    await say("09123333434")
    llm.answers = [{"reply": "آدرس؟", "intent": "order", "order_action": "update", "order": {"name": "علی"}}]
    m = await say("علی")
    assert "خلاصه" not in m["text"]
    llm.answers = [{"reply": "ممنون", "intent": "order", "order_action": "update", "order": {"address": "قم صفاییه"}}]
    m = await say("قم صفاییه")
    assert "خلاصهٔ سفارشت" in m["text"] and "👤 علی" in m["text"] and "📍 قم صفاییه" in m["text"]
    assert all("STEP" not in c["prompt"] for c in llm.calls[:-1])


async def test_order_tracking_code_and_status_from_the_orders(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550036")

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    # no order yet: a pre-purchase delivery question still goes to the assistant
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    await say("سلام")
    n = len(llm.calls)
    await say("اگه الان خرید کنم کی میرسه؟")
    assert len(llm.calls) == n + 1
    llm.answers = [{"reply": "", "intent": "order_status"}]
    m = await say("ی سفارش قبلا زدم وضعیتش چیه؟")
    assert m["source"] == "status" and "کد رهگیری" in m["text"]
    assert "Never say an order is placed" in llm.calls[-1]["system"]

    await _order_ready(client, h, sid, llm, p1)
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    m = await say("بله")
    code = re.search(r"S\d{7}", m["text"]).group()
    n = len(llm.calls)
    llm.answers = [{"reply": "", "intent": "order_status"}]
    m = await say("وضعیت سفارشم رو بگو کجاس")
    assert m["source"] == "status" and "منتظر پرداخت" in m["text"] and code in m["text"] and "مانتو لینن کرم" in m["text"]
    assert "آبان" in m["text"] or "۱۴۰" in m["text"]
    assert len(llm.calls) == n + 1
    # sending the tracking code while the receipt is awaited is a lookup, not a payment reference
    m = await say(code)
    assert m["source"] == "status"
    assert (await say("کد پیگیری پرداخت ۵۵۱۲۰۹۸۸"))["source"] == "receipt"
    llm.answers = [{"reply": "", "intent": "order_status"}]
    assert "فیش رسیده" in (await say("سفارشم چی شد؟"))["text"]
    # a follow-up ("why?") is answered by the model from the orders it sees, not with the same list again
    llm.answers = [{"reply": "فیشت رسیده و ادمین داره بررسیش می‌کنه 🌸", "intent": "order_status"}]
    m = await say("چرا هنوز تایید نشده؟")
    assert m["text"] == "فیشت رسیده و ادمین داره بررسیش می‌کنه 🌸" and "📦" not in m["text"]
    assert "MY ORDERS" in llm.calls[-1]["prompt"] and code in llm.calls[-1]["prompt"]

    # a new chat finds it only by the tracking code
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"reset": True})
    await say("سلام")
    llm.answers = [{"reply": "", "intent": "order_status"}]
    m = await say("وضعیت سفارشم")
    assert "S1234567" in m["text"]
    m = await say(f"کدش {code}")
    assert "فیش رسیده" in m["text"]
    m = await say("S0000000")
    assert "پیدا نکردم" in m["text"]
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert rows[0]["tracking_code"] == code


async def _notes(sid):
    from sqlalchemy import select
    from app.core.db import SessionLocal
    from app.models import Notification
    async with SessionLocal() as db:
        return [n.title_fa for n in (await db.execute(select(Notification).where(Notification.store_id == sid)
                                                      .order_by(Notification.id))).scalars()]


async def test_second_order_after_a_confirmed_one_runs_the_order_steps(client, monkeypatch):
    """ali's chat: order #1 paid and confirmed by the seller, then "a hoodie too, 2XL, you choose": the model wrote
    «ثبت شد» and nothing was saved. Now the summary (with the earlier contact details) and the yes create order #2."""
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550041")

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await _order_ready(client, h, sid, llm, p1)
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    assert (await say("بله"))["source"] == "order"
    assert (await say("فیش ۱۲۸۲۹۲۹۸۲۸۲۹۲"))["source"] == "receipt"
    rows = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    r = await client.patch(f"/ai-orders/{rows[0]['id']}", headers=h, json={"status": "confirmed"})
    assert r.status_code == 200, r.text

    llm.answers = [{"reply": "ست هودی رو براتون در نظر گرفتم. سایز ۲XL ثبت شد.", "intent": "order",
                    "order_action": "update", "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "2XL"}]}}]
    m = await say("خودت انتخاب کن")
    assert "خلاصهٔ سفارشت" in m["text"] and "ثبت شد" not in m["text"]
    assert "سارا کریمی" in m["text"] and "شیراز" in m["text"] and "2XL" in m["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    m = await say("بله")
    assert m["source"] == "order" and "ثبت شد" in m["text"] and "6037991234567890" in m["text"]
    rows = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    assert len(rows) == 2

    # "ثبت شد" with no item at all is not left standing either
    llm.answers = [{"reply": "سفارشتون ثبت شد.", "intent": "order"}]
    assert "ثبت شد" not in (await say("یه دونه دیگه هم میخوام"))["text"]


async def test_a_new_customer_whose_order_the_model_claims_gets_asked_for_details(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550042")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "سایز L ثبت شد.", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": p1["id"], "qty": 1, "variant": "L"}]}}]
    m = await say("L بزار برام")
    assert "ثبت شد" not in m["text"] and "نام و نام خانوادگی" in m["text"] and "آدرس کامل" in m["text"]
    llm.answers = [{"reply": "ممنون 🌸", "intent": "order", "order_action": "update", "order": {
        "name": "سید علی هانی", "phone": "۰۹۱۲۳۳۳۴۴۵۵", "address": "قم دورشهر کوچه ۲ کد پستی هم ۲۲۲۳۳۴۵۶۱۲"}}]
    m = await say("سید علی هانی\n۰۹۱۲۳۳۳۴۴۵۵\nقم دورشهر کوچه ۲ کد پستی هم ۲۲۲۳۳۴۵۶۱۲")
    assert "خلاصهٔ سفارشت" in m["text"] and "۲۲۲۳۳۴۵۶۱۲" in m["text"] and "09123334455" in m["text"]


async def test_swearing_is_reported_and_complaints_go_to_the_admin(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550043")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "", "intent": "abuse"}, {"reply": "", "intent": "abuse"}]
    m = await say("مرتیکه کسکش جواب بده")
    assert m["intent"] == "abuse" and "محترمانه" in m["text"]
    await say("کثافت")  # a second one within minutes: answered, not reported again
    notes = await _notes(sid)
    assert sum("فحاشی" in t for t in notes) == 1

    # a complaint the FAQ can't settle: the admin is told and takes the chat
    llm.answers = [{"reply": "متاسفم", "intent": "handoff", "handoff": True, "handoff_reason": "complaint"}]
    m = await say("دو هفته‌ست منتظرم هیچکس جواب درست نمیده")
    assert m["source"] == "handoff" and "ادمین" in m["text"]
    assert any("شکایت" in t for t in await _notes(sid))
    conv = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert conv["state"] is not None


async def test_a_long_faq_is_sent_a_few_entries_at_a_time(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550044")
    faqs = [{"q": f"سوال شماره {i} دربارهٔ موضوع {i}", "a": "جواب طولانی " * 20} for i in range(12)]
    faqs.append({"q": "ارسال با چی انجام میشه؟", "a": "ارسال فقط با تیپاکس، دوشنبه و چهارشنبه."})
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True, "faqs": faqs})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    await say("کی برام ارسال میشه به دستم میرسه")
    call = llm.calls[-1]
    assert "تیپاکس" in call["prompt"] and "تیپاکس" not in call["system"]
    assert "جواب طولانی" not in call["system"] and call["prompt"].count("جواب طولانی جواب") <= chat_engine.FAQ_PER_TURN


def test_variant_options():
    p = chat_engine.Product(attributes={"سایز": "L، XL، XXL", "رنگ": "مشکی و سفید"})
    o = chat_engine.variant_options(p)
    assert o == {"size": ["L", "XL", "XXL"], "color": ["مشکی", "سفید"]}


async def test_every_message_goes_to_the_model_first(client, monkeypatch):
    """ali: «the last message never reached the AI». Product questions, "I want to order", a bare size, hi and
    thanks are all read by the model; the code only carries out what it decides."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550045")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"},
        "faqs": [{"q": "هزینه ارسال چقدره؟", "a": "۷۰ هزار"}]})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    texts = ("سایز بندی رو لطف میکنی؟", "موجوده؟", "خب میخوام سفارش بدم", "۴۰", "سلام", "مرسی", "هزینه ارسال چقدره؟",
             "می‌خوام با ادمین صحبت کنم", "چه رنگایی داری", "قهوه‌ای رو میخوام")
    for i, t in enumerate(texts, 1):
        m = await say(t)
        assert len(llm.calls) == i and llm.calls[-1]["prompt"].endswith(f"<customer>{t}</customer>"), t
        assert m["source"] == "llm"


async def test_order_edits_at_the_summary_are_read_by_the_model(client, monkeypatch):
    """ali's chats: «نه دیگه… قهوه‌ای میخوام» at the summary cancelled the order, a removed item stayed in the summary,
    and the product code was used as the id. Now the model reads the answer to the summary and the code follows it."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550046")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    brown = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "هودی مردانه قهوه‌ای با جزئیات دوخت برجسته", "price": 2000000, "stock_qty": 3})).json()
    kapli = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "هودی کاپلی مشکی", "price": 2200000, "stock_qty": 3, "attributes": {"سایز": "2XL"}})).json()
    await client.patch(f"/products/{kapli['id']}", headers=h, json={"sku": str(brown["id"])})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "هودی کاپلی و هودی قهوه‌ای داریم. کدوم؟", "intent": "clarify",
                    "products": [kapli["id"], brown["id"]]}]
    await say("برام یه هودی بیار")
    # the model used the product's code (= the brown hoodie's id) for «کاپلی»: the chat's words fix it
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm",
                    "order": {"items": [{"product_id": brown["id"], "qty": 1, "variant": "2XL"}],
                              "name": "هانی", "phone": "09124443322", "address": "قم خیابان صفاییه 42 1"}}]
    m = await say("کاپلی، هانی 09124443322 قم خیابان صفاییه 42 1")
    assert "خلاصهٔ سفارشت" in m["text"] and "کاپلی" in m["text"] and "قهوه" not in m["text"]

    # a plain question at the summary: answered, the summary still waits
    llm.answers = [{"reply": "ارسال ۲ تا ۳ روز کاریه 🌸", "intent": "faq"}]
    m = await say("کی میرسه؟")
    assert m["text"] == "ارسال ۲ تا ۳ روز کاریه 🌸" and "STEP: summary" in llm.calls[-1]["prompt"]

    # «نه، …» with a change is a change, not a cancel: a new summary
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": kapli["id"], "qty": 1, "variant": "2XL"},
                                        {"product_id": brown["id"], "qty": 1}], "phone": "09351112233"}}]
    m = await say("نه صبر کن، قهوه‌ای هم میخوام، شمارمم 09351112233")
    assert "STEP: summary" in llm.calls[-1]["prompt"]
    assert "خلاصهٔ سفارشت" in m["text"] and "قهوه" in m["text"] and "09351112233" in m["text"] and "لغو" not in m["text"]

    # removing an item takes it off the summary
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": brown["id"], "qty": 0}]}}]
    m = await say("قهوه‌ای رو حذف کن")
    assert "خلاصهٔ سفارشت" in m["text"] and "قهوه" not in m["text"] and "کاپلی" in m["text"]

    # the model says yes, re-sending the same order in other words: placed, not shown again
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm",
                    "order": {"items": [{"product_id": kapli["id"], "qty": 1, "variant": " 2XL "}]}}]
    m = await say("بله")
    assert m["source"] == "order" and "ثبت شد" in m["text"]
    rows = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    assert [i["product_id"] for i in rows[0]["items"]] == [kapli["id"]] and rows[0]["customer_phone"] == "09351112233"


async def test_cancel_at_the_summary_and_the_addition(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550047")
    r = await _order_ready(client, h, sid, llm, p1)
    assert "خلاصهٔ سفارشت" in r["messages"][-1]["text"]

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    llm.answers = [{"reply": "باشه، کدوم قسمت رو عوض کنم؟", "intent": "order"}]
    assert "کدوم قسمت" in (await say("نه"))["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "cancel"}]
    assert "لغو شد" in (await say("کلا نمیخوام لغو"))["text"]
    conv = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert not (conv["state"] or {}).get("order") and not (conv["state"] or {}).get("awaiting")


async def test_wrong_length_phones_and_fixing_a_placed_order(client, monkeypatch):
    """ali's «هودی اختاپوس» chat: a 10-digit phone, a 12-digit one, and the corrected phone after the order."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550048")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "ممنون", "intent": "order", "order_action": "confirm", "order": {
        "items": [{"product_id": p1["id"], "qty": 1, "variant": "۴۰"}], "name": "jsj", "phone": "0932453454",
        "address": "fsd fdg"}}]
    m = await say("مانتو ۴۰، jsj، 0932453454، fsd fdg")
    assert "0932453454" in m["text"] and "۱۰ رقمیه" in m["text"] and "خلاصه" not in m["text"]
    llm.answers = [{"reply": "ممنون", "intent": "order", "order_action": "update", "order": {"phone": "092452435254"}}]
    m = await say("092452435254")
    assert "۱۲ رقمیه" in m["text"] and "خلاصه" not in m["text"]
    llm.answers = [{"reply": "ممنون", "intent": "order", "order_action": "update", "order": {"phone": "09124443322"}}]
    m = await say("09124443322")
    assert "خلاصهٔ سفارشت" in m["text"] and "09124443322" in m["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    assert "ثبت شد" in (await say("اره"))["text"]

    # after the order: a mobile number is never filed as a payment reference; the model reads it as a correction
    llm.answers = [{"reply": "", "intent": "order", "order_action": "update", "order": {"phone": "092343535435"}}]
    m = await say("092343535435")
    assert m["source"] != "receipt" and "۱۲ رقمیه" in m["text"]
    llm.answers = [{"reply": "", "intent": "order", "order_action": "update", "order": {"phone": "09351112233"}}]
    m = await say("09351112233")
    assert m["source"] != "receipt" and "اصلاح شد" in m["text"] and "09351112233" in m["text"]
    assert (await say("کد پیگیری ۷۷۸۸۹۹۰۰"))["source"] == "receipt"
    rows = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = rows.get("items", rows) if isinstance(rows, dict) else rows
    assert rows[0]["customer_phone"] == "09351112233" and rows[0]["customer_name"] == "jsj"
    assert any("شماره موبایل سفارش" in t for t in await _notes(sid))


def test_phone_numbers_of_the_wrong_length_are_not_cut_down():
    for bad in ("0932453454", "092452435254", "0912 345 678"):
        assert chat_engine.norm_phone(bad) == "" and chat_engine.bad_phone(bad), bad
    for good, want in (("0912 345 6789", "09123456789"), ("+98 912 345 6789", "09123456789"),
                       ("989335137843", "09335137843"), ("۰۹۱۲۳۴۵۶۷۸۹", "09123456789")):
        assert chat_engine.norm_phone(good) == want, good


async def test_a_product_in_another_colour_is_not_put_in_the_cart(client, monkeypatch):
    """ali's chat: "black hoodie with my design" — the model put the brown hoodie in the cart as «مشکی، M»."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550049")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    brown = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "هودی کلاهدار قهوه‌ای", "price": 2000000, "stock_qty": 3, "attributes": {"سایز": "M، L، XL"}})).json()

    async def say(text):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()["messages"][-1]

    await say("سلام")
    llm.answers = [{"reply": "باشه، اضافه کردم.", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": brown["id"], "qty": 1, "variant": "مشکی، سایز: M"}]}}]
    m = await say("هودی مشکی سایز مدیوم میخوام")
    assert "رنگ مشکی نداره" in m["text"] and "هودی کلاهدار قهوه‌ای" in m["text"]
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": brown["id"], "qty": 1, "variant": "3XL"}]}}]
    m = await say("سه ایکس بده")
    assert "سایز 3XL نداره" in m["text"] and "M، L، XL" in m["text"]
    conv = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert not ((conv["state"] or {}).get("order") or {}).get("items")
    # the colour it comes in is fine
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": brown["id"], "qty": 1, "variant": "قهوه ای، دوایکس"}]}}]
    await say("قهوه‌ای XL")
    conv = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert not ((conv["state"] or {}).get("order") or {}).get("items")  # 2XL isn't one of its sizes either
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": brown["id"], "qty": 1, "variant": "قهوه ای، L"}]}}]
    await say("قهوه‌ای L")
    conv = (await client.get(f"/stores/{sid}/chat/test", headers=h)).json()
    assert conv["state"]["order"]["items"][0]["product_id"] == brown["id"]


async def test_the_customers_photo_is_read_and_a_design_goes_on_the_order(client, monkeypatch):
    """ali's chat: the design photos got «عکس رسید، اسم محصول رو بنویس» and the model kept asking for the design."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550050")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    from app.services import media
    monkeypatch.setattr(media, "read_local", lambda url: (b"img", "image/png") if url.startswith("/media/") else None)
    seen: list[dict] = []

    async def vision(*, task, system, prompt, images=None, store_id=None, **kw):
        seen.append({"task": task, "prompt": prompt, "images": images})
        return {"kind": "design", "description": "نقاشی یک گربه‌ی سیاه با متن ABRA", "product_id": 0}
    monkeypatch.setattr(chat_engine.ai, "generate_json", vision)

    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    from app.core.db import SessionLocal
    from sqlalchemy import select
    async with SessionLocal() as db:
        conv = (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first()
        rt.add_message(db, conv, role="customer", text="", image_url="/media/1/design.jpg")
        await db.commit()
        cid = conv.id
    llm.answers = [{"reply": "طرح گربه رو گرفتم 🌸 روی کدوم هودی و چه سایزی چاپ کنیم؟", "intent": "order"}]
    saved = await rt.answer(cid, send=False)
    assert seen and seen[0]["task"] == "chat_vision" and seen[0]["images"] and "مانتو لینن کرم" in seen[0]["prompt"]
    assert "گربه‌ی سیاه" in llm.calls[-1]["prompt"] and "طرح برای چاپ" in llm.calls[-1]["prompt"]
    assert saved[-1].text.startswith("طرح گربه رو گرفتم")
    async with SessionLocal() as db:
        conv = await db.get(chat_engine.ChatConversation, cid)
        assert conv.state["pending_design"] == "/media/1/design.jpg"

    # no vision model (or an unreadable photo): the old fixed answer, no chat model call
    async def broken(**kw):
        raise chat_engine.ai.AIError("x")
    monkeypatch.setattr(chat_engine.ai, "generate_json", broken)
    n = len(llm.calls)
    async with SessionLocal() as db:
        rt.add_message(db, await db.get(chat_engine.ChatConversation, cid), role="customer", text="",
                       image_url="/media/1/other.jpg")
        await db.commit()
    saved = await rt.answer(cid, send=False)
    assert saved[-1].text == chat_engine.PHOTO_TEXT and len(llm.calls) == n


async def _print_shop(client, monkeypatch, phone):
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, phone)
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    hoodie = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "هودی چاپ طرح دلخواه", "price": 900000, "stock_qty": 20,
        "attributes": {"رنگ": "مشکی، سفید", "سایز": "M، L، XL", "محل چاپ": "جلو، پشت"}})).json()
    from app.services import media
    monkeypatch.setattr(media, "read_local", lambda url: (b"img", "image/png") if url.startswith("/media/") else None)

    async def vision(*, task, system, prompt, images=None, store_id=None, **kw):
        return {"kind": "product", "description": "عکس اسپایدرمن", "product_id": 0}
    monkeypatch.setattr(chat_engine.ai, "generate_json", vision)
    return h, sid, llm, hoodie


async def _send_photo(client, h, sid, url):
    from app.core.db import SessionLocal
    from sqlalchemy import select
    async with SessionLocal() as db:
        conv = (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first()
        rt.add_message(db, conv, role="customer", text="", image_url=url)
        await db.commit()
        return conv.id


async def _state(sid):
    from app.core.db import SessionLocal
    from sqlalchemy import select
    async with SessionLocal() as db:
        return (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first().state


async def test_print_order_keeps_each_design_asks_the_side_and_places_only_on_yes(client, monkeypatch):
    """ali's chat with order #5: the designs were lost, «جلو» was ignored, «یه سفارش دیگه هم میخوام» placed the
    order and «ایتم‌هام رو ببینم» tripled it."""
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550060")
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    cid = await _send_photo(client, h, sid, "/media/1/spider.jpg")
    llm.answers = [{"reply": "طرح رسید 🌸", "intent": "order"}]
    await rt.answer(cid, send=False)
    assert (await _state(sid))["pending_design"] == "/media/1/spider.jpg"

    # all fields but no print side: it is asked, not confirmed
    order = {"items": [{"product_id": hoodie["id"], "qty": 1, "variant": "مشکی L"}], "name": "سارا کریمی",
             "phone": "09171234567", "address": "شیراز، معالی آباد"}
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": order}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "هودی مشکی L"})).json()
    assert "محل چاپ" in r["messages"][-1]["text"] and "جلو" in r["messages"][-1]["text"]
    st = await _state(sid)
    assert st["order"]["items"][0]["design"] == "/media/1/spider.jpg" and "pending_design" not in st

    order["items"][0]["variant"] = "مشکی L جلو"
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": order}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "جلو"})).json()
    assert "خلاصهٔ سفارشت" in r["messages"][-1]["text"] and "🎨 طرح رسید" in r["messages"][-1]["text"]

    # not a yes: nothing is placed
    llm.answers = [{"reply": "حتماً", "intent": "order", "order_action": "confirm"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "یه سفارش دیگه هم میخوام"})).json()
    assert r["messages"][-1]["text"].endswith(chat_engine.NOT_PLACED_TEXT)
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    assert not (orders.get("items", orders) if isinstance(orders, dict) else orders)

    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm"}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "بله"})).json()
    assert r["messages"][-1]["source"] == "order"
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert len(rows) == 1 and list(rows[0]["extras"].values()) == ["/media/1/spider.jpg"]
    assert "طرح آیتم ۱" in list(rows[0]["extras"])[0]

    # the model puts the same items in again («ایتم‌هام رو ببینم»): nothing is added
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": order}]
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "ایتم هام رو ببینم"})).json()
    assert "توی سفارشت هستن" in r["messages"][-1]["text"]
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert len(rows) == 1 and sum(i["qty"] for i in rows[0]["items"]) == 1


def test_plain_yes():
    for t in ("بله", "آره ثبت کن", "اوکی", "تایید", "بله درسته"):
        assert chat_engine.plain_yes(t), t
    for t in ("یه سفارش دیگه هم میخوام", "نه", "بله ولی سایزش رو عوض کن", "چرا ۶ تا؟", "آیتم هام رو ببینم"):
        assert not chat_engine.plain_yes(t), t


async def test_size_chart_image_is_read_once(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550061")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    from app.core.db import SessionLocal
    from app.models import ChatReference
    async with SessionLocal() as db:
        db.add(ChatReference(store_id=sid, title="جدول سایز", image_url="/media/1/chart.jpg"))
        await db.commit()
    from app.services import media
    monkeypatch.setattr(media, "read_local", lambda url: (b"img", "image/png"))
    calls = []

    async def vision(*, task, system, prompt, images=None, store_id=None, **kw):
        calls.append(task)
        return {"text": "M: عرض 56، قد 72"}
    monkeypatch.setattr(chat_engine.ai, "generate_json", vision)
    llm.answers = [{"reply": "سلام", "intent": "greeting"}, {"reply": "عرض M 56 هست", "intent": "product"}]
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "عرض سایز M چنده؟"})
    assert calls == ["chat_vision"] and "M: عرض 56، قد 72" in llm.calls[-1]["system"]


async def test_spiderman_chat_photo_cart_and_other_designs(client, monkeypatch):
    """ali's chat: the promised photo never came, «سایز دوایکس رو میخوام رنگ مشکی» left the cart empty, and
    «دیگه چه مدلایی از اسپایدرمن موجوده؟» got the whole catalog."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550062")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True, "payment": {"card_number": "6037991234567890"}})
    from app.core.db import SessionLocal
    from app.models import Product, ProductImage
    async with SessionLocal() as db:
        spider = Product(store_id=sid, title="ست هودی اسپایدرمن", price=2000000, stock_qty=5, category="هودی",
                         attributes={"رنگ": "مشکی", "سایز": "L، XL، 2XL"})
        octo = Product(store_id=sid, title="هودی اختاپوس", price=2000000, stock_qty=1, category="هودی")
        jean = Product(store_id=sid, title="شلوار جین مام فیت", price=3000000, stock_qty=1, category="شلوار")
        db.add_all([spider, octo, jean])
        await db.flush()
        db.add(ProductImage(product_id=spider.id, url="/media/x/spider.jpg"))
        await db.commit()
        sp, oc = spider.id, octo.id

    async def say(t):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": t})).json()["messages"]

    await say("سلام")
    llm.answers = [{"reply": "سلام! ست هودی اسپایدرمن رو داریم که رنگ مشکی داره.", "intent": "product",
                    "products": [sp]}]
    await say("هودی مشکی اسپایدرمن میخوام")
    llm.answers = [{"reply": "حتما، عکس ست هودی اسپایدرمن رو براتون می‌فرستم.", "intent": "product"}]
    msgs = await say("ببینم عکس خود کاررو")
    assert "/media/x/spider.jpg" in [m["image_url"] for m in msgs]

    llm.answers = [{"reply": "حتما، سایز 2XL رنگ مشکی موجود هست.", "intent": "product"}]
    await say("سایز دوایکس رو میخوام رنگ مشکی")
    st = await _state(sid)
    assert st["order"]["items"][0]["product_id"] == sp and "2XL" in st["order"]["items"][0]["variant"]

    llm.answers = [{"reply": "", "intent": "catalog"}]
    last = (await say("یه طرح دیگه میتونی بهم نشون بدی؟دیگه چه مدلایی از اسپایدرمن موجوده؟"))[-1]["text"]
    assert "فقط همین «ست هودی اسپایدرمن»" in last and "هودی اختاپوس" in last and "شلوار" not in last

    llm.answers = [{"reply": "", "intent": "catalog"}]
    last = (await say("هودی بتمن دارین؟"))[-1]["text"]
    assert "نداریم" in last and "هودی اختاپوس" in last and "جین" not in last
    assert oc


async def test_second_spiderman_chat(client, monkeypatch):
    """ali's second try: a size-chart question became the summary, «بشت»/«پشت» never reached the order, «ثبت شد» and
    «در حال ارساله» were said of a cart, and «عکسش رو نداریم» came right under the photo."""
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550063")
    from app.core.db import SessionLocal
    from app.models import Product, ProductImage
    async with SessionLocal() as db:
        spider = Product(store_id=sid, title="ست هودی اسپایدرمن", price=2000000, stock_qty=5, category="هودی",
                         attributes={"رنگ": "مشکی", "سایز": "L، XL، 2XL"})
        db.add(spider)
        await db.flush()
        db.add(ProductImage(product_id=spider.id, url="/media/x/spider.jpg"))
        await db.commit()
        sp = spider.id

    async def say(t):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": t})).json()["messages"]

    await say("سلام")
    llm.answers = [{"reply": "ست هودی اسپایدرمن رو داریم 🌸", "intent": "product", "products": [sp]}]
    await say("ست هودی اسپایدرمن دارین؟")
    llm.answers = [{"reply": "عکس خود کار رو نداریم، ولی می‌تونم عکس ست هودی اسپایدرمن رو براتون بفرستم.",
                    "intent": "product"}]
    msgs = await say("ببینم عکس خود کاررو")
    assert msgs[-1]["image_url"] == "/media/x/spider.jpg" and "نداریم" not in msgs[-1]["text"]

    llm.answers = [{"reply": "قد و عرض 2XL توی جدول نوشته نشده 🙏", "intent": "product"}]
    msgs = await say("دوایکس داخل جدول ابعادش چجوریه قد و عرض کار؟")
    assert msgs[-1]["text"].startswith("قد و عرض") and not (await _state(sid)).get("order")

    # the print hoodie with its design, then the print side in a typo
    cid = await _send_photo(client, h, sid, "/media/1/mine.jpg")
    llm.answers = [{"reply": "طرح رسید 🌸", "intent": "order"}]
    await rt.answer(cid, send=False)
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm", "order": {
        "items": [{"product_id": hoodie["id"], "qty": 1, "variant": "مشکی XL"}], "name": "علی هانی",
        "phone": "09121112233", "address": "قم دورشهر کوچه ۲"}}]
    assert "محل چاپ" in (await say("ثبت کنید برام"))[-1]["text"]
    llm.answers = [{"reply": "چاپ پشت هودی ثبت شد.", "intent": "order", "order_action": "update"}]
    last = (await say("بشت"))[-1]["text"]
    assert "خلاصهٔ سفارشت" in last and "پشت" in last and "ثبت شد" not in last

    # "when do you send it?" of a cart: never "it's on its way"
    llm.answers = [{"reply": "سفارشتون در حال آماده‌سازی و ارساله.", "intent": "order_status"}]
    last = (await say("خب کی میفرستی"))[-1]["text"]
    assert "در حال" not in last and ("خلاصهٔ سفارشت" in last or "ثبت نشده" in last)


async def test_cart_and_order_buttons(client, monkeypatch):
    """The two fixed buttons, the summary's yes / change, and changing an option from the product's own choices:
    answered by code (no model call), with the keyboard on the bot's last message."""
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550064")
    from app.services import chat_menu as menu
    scheduled: list[int] = []
    monkeypatch.setattr(rt, "schedule", lambda cid: scheduled.append(cid))
    bot = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "445:menu-token"})).json()
    n = 0

    async def tap(text, **extra):
        nonlocal n
        n += 1
        await _customer(bot["id"], 2002, text, n, **extra)
        return await rt.answer(scheduled[-1])

    await tap("/start")
    SENT.clear()
    saved = await tap(menu.CART)
    assert saved[-1].text == menu.EMPTY and not llm.calls
    markup = SENT[-1][1]["reply_markup"]
    assert markup["keyboard"] == [[{"text": menu.CART}, {"text": menu.ORDER}]] and markup["is_persistent"]
    assert (await tap(menu.ORDER))[-1].text == menu.EMPTY_ORDER

    # the model puts the print hoodie in the cart, with no print side or design yet
    llm.answers = [{"reply": "گذاشتمش توی سبد 🌸", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": hoodie["id"], "qty": 1, "variant": "مشکی L"}]}}]
    await tap("هودی چاپی مشکی L میخوام")
    calls = len(llm.calls)
    saved = await tap(menu.ORDER)
    assert "محل چاپ" in saved[-1].text and saved[-1].buttons == [["جلو", "پشت"]]
    assert SENT[-1][1]["reply_markup"]["one_time_keyboard"]
    saved = await tap("پشت")
    assert saved[-1].text.startswith(menu.DONE) and "طرحی که می‌خوای چاپ بشه" in saved[-1].text
    assert len(llm.calls) == calls

    # the design photo goes the usual way (the model reads it), then the order button asks the details one by one
    llm.answers = [{"reply": "طرح رسید 🌸", "intent": "order"}]
    from app.services import media
    assert media
    monkeypatch.setattr(rt, "save_bytes", lambda store_id, data, name: "/media/1/d.jpg")
    monkeypatch.setattr(rt.tg, "download", lambda bot, fid: _async((b"img", "d.jpg")))
    await tap("", photo=[{"file_id": "f1"}])
    from app.core.db import SessionLocal
    async with SessionLocal() as db:
        conv = await db.get(chat_engine.ChatConversation, scheduled[-1])
        assert conv.state["order"]["items"][0].get("design") == "/media/1/d.jpg"
    assert (await tap(menu.ORDER))[-1].text == menu.ASK_FIELD["name"]
    saved = await tap("سارا کریمی")
    assert saved[-1].text == menu.ASK_FIELD["phone"] and saved[-1].buttons == [[menu.SEND_PHONE]]
    assert SENT[-1][1]["reply_markup"]["keyboard"] == [[{"text": menu.SEND_PHONE, "request_contact": True}]]
    assert "درست نیست" in (await tap("0912"))[-1].text
    await tap("", contact={"phone_number": "+989171234567"})
    saved = await tap("شیراز، معالی آباد، کوچه ۳، کد پستی ۱۲۳۴۵")
    assert saved[-1].text.startswith("خلاصهٔ سفارشت") and saved[-1].buttons == menu.SUMMARY_BUTTONS
    assert "پشت" in saved[-1].text and "09171234567" in saved[-1].text

    # change → options → colour, from the product's own choices
    assert (await tap(menu.CHANGE))[-1].buttons[0] == [menu.CH_OPTS, menu.CH_QTY]
    saved = await tap(menu.CH_OPTS)
    assert saved[-1].text.startswith("برای «هودی چاپ طرح دلخواه» کدوم") and saved[-1].buttons == [["رنگ", "سایز", "محل چاپ"]]
    saved = await tap("رنگ")
    assert saved[-1].buttons == [["✔️ مشکی", "سفید"]]
    saved = await tap("سفید")
    assert saved[-1].text.startswith(menu.DONE) and "(سفید L پشت)" in saved[-1].text
    await tap(menu.CHANGE)
    await tap(menu.CH_QTY)
    saved = await tap("۲")
    assert "× ۲" in saved[-1].text and len(llm.calls) == calls + 1

    # yes places it, with the payment text
    saved = await tap(menu.YES)
    assert saved[-1].source == "order" and "6037991234567890" in saved[-1].text
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert rows[0]["items"][0]["variant"] == "سفید L پشت" and rows[0]["items"][0]["qty"] == 2
    assert (await tap(menu.CART))[-1].buttons == [[menu.DETAILS]]
    assert "#" in (await tap(menu.DETAILS))[-1].text


async def _async(v):
    return v


async def test_cart_buttons_cancel_and_typing(client, monkeypatch):
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550065")
    from app.services import chat_menu as menu

    async def say(t):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": t})).json()["messages"]

    await say("سلام")
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": hoodie["id"], "qty": 1, "variant": "سفید M جلو"}]}}]
    await say("هودی چاپی سفید M جلو")
    msgs = await say(menu.CART)
    assert "🛒 سبد خریدت" in msgs[-1]["text"] and msgs[-1]["buttons"] == [[menu.ORDER], [menu.EDIT_CART, menu.CLEAR]]
    await say(menu.EDIT_CART)
    # typing instead of tapping leaves the step: the model answers
    llm.answers = [{"reply": "ارسال ۳ روزه 🌸", "intent": "faq"}]
    assert (await say("ارسال چند روزه؟"))[-1]["text"] == "ارسال ۳ روزه 🌸"
    assert "menu" not in (await _state(sid))
    await say(menu.CHANGE)
    await say(menu.CH_CANCEL)
    assert (await say(menu.YES_CANCEL))[-1]["text"] == menu.CANCELLED
    assert "order" not in (await _state(sid))


async def test_third_spiderman_chat(client, monkeypatch):
    """ali's third try: an old order's details jumped to a summary, a design photo was taken as order #6's receipt,
    «سایز رو بفرمایید» repeated with the size in the cart, a yes tapped under an older summary, «مدل‌های دیگه»
    answered vaguely, and «هودی وارداتی» came up as Spider-Man."""
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550066")
    from app.core.db import SessionLocal
    from app.models import AiOrder, Product, ProductImage
    from app.services import chat_menu as menu
    async with SessionLocal() as db:
        spider = Product(store_id=sid, title="ست هودی اسپایدرمن", price=2000000, stock_qty=5, category="هودی",
                         attributes={"رنگ": "مشکی"})
        imported = Product(store_id=sid, title="هودی وارداتی", price=2000000, stock_qty=5, category="هودی",
                           tags=["اسپ"])
        db.add_all([spider, imported])
        await db.flush()
        db.add(ProductImage(product_id=spider.id, url="/media/x/spider.jpg"))
        await db.commit()
        sp = spider.id

    async def say(t):
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": t})).json()["messages"]

    await say("سلام")
    async with SessionLocal() as db:
        conv = (await db.execute(select(chat_engine.ChatConversation).where(
            chat_engine.ChatConversation.store_id == sid, chat_engine.ChatConversation.is_test))).scalars().first()
        db.add(AiOrder(store_id=sid, conversation_id=conv.id, customer_name="سید علی هانی", customer_phone="09233333333",
                       customer_address="قم دورشهر کوچه ۲", items=[], status="confirmed", is_test=True))
        await db.commit()

    # 1. wanting it is not yet the summary, even with an old order's details at hand
    llm.answers = [{"reply": "ست هودی اسپایدرمن مشکی رو داریم 🌸", "intent": "product", "products": [sp]}]
    assert "خلاصهٔ سفارشت" not in (await say("هودی مشکی اسپایدرمن میخوام"))[-1]["text"]

    # 2. "other Spider-Man models?" read by the model as a product chat: the shop's search answers
    llm.answers = [{"reply": "سلام! بله حتماً. طرح‌های دیگه رو توی محصولاتمون ببینید.", "intent": "product"}]
    last = (await say("یه طرح دیگه میتونی بهم نشون بدی؟دیگه چه مدلایی از اسپایدرمن موجوده؟"))[-1]["text"]
    assert "فقط همین «ست هودی اسپایدرمن»" in last and "وارداتی" not in last.split("داریم:")[0]

    # 3. a yes tapped under an older summary shows the new one first
    llm.answers = [{"reply": "", "intent": "order", "order_action": "confirm",
                    "order": {"items": [{"product_id": sp, "qty": 1, "variant": "مشکی"}]}}]
    assert "خلاصهٔ سفارشت" in (await say("همین رو ثبت کن"))[-1]["text"]
    from datetime import timedelta
    async with SessionLocal() as db:
        conv = await db.get(chat_engine.ChatConversation, conv.id)
        conv.state = {**conv.state, "summary_at": (chat_engine.utcnow() + timedelta(seconds=30)).isoformat()}
        await db.commit()
    assert (await say(menu.YES))[-1]["text"].startswith("این خلاصه عوض شده")
    msgs = await say(menu.YES)
    assert msgs[-1]["source"] == "order"

    # 4. waiting for order's receipt, the customer sends a design for a new print hoodie: not a receipt
    llm.answers = [{"reply": "طرح رسید 🌸 روی چه رنگ و سایزی؟", "intent": "order"}]
    cid = await _send_photo(client, h, sid, "/media/1/mine.jpg")

    async def vision(*, task, system, prompt, images=None, store_id=None, **kw):
        return {"kind": "design", "description": "طرح دلخواه", "product_id": 0}
    monkeypatch.setattr(chat_engine.ai, "generate_json", vision)
    saved = await rt.answer(cid, send=False)
    assert "دریافت شد" not in saved[-1].text and (await _state(sid))["pending_design"] == "/media/1/mine.jpg"
    orders = (await client.get(f"/stores/{sid}/ai-orders", headers=h, params={"include_test": True})).json()
    rows = orders.get("items", orders) if isinstance(orders, dict) else orders
    assert all(o["status"] != "payment_submitted" for o in rows)

    # 5. the model asks for the size the cart already has: the shop's next step instead
    llm.answers = [{"reply": "باشه، پشت چاپ میشه. سایز رو هم لطفاً بفرمایید.", "intent": "order",
                    "order_action": "update", "order": {"items": [
                        {"product_id": hoodie["id"], "qty": 1, "variant": "مشکی XL پشت"}]}}]
    last = (await say("همین طرح رو روی هودی مشکی XL میخوام چاپش پشت"))[-1]["text"]
    assert "سایز رو هم" not in last and ("اضافه به سفارش" in last or "خلاصهٔ سفارشت" in last)


async def test_router_sends_each_message_to_its_skill(client, monkeypatch):
    """ali's idea: each message is first sorted into its part of the journey; the shop's own parts answer without
    the sales model, the rest go to it with the part named, and an unsure router changes nothing."""
    h, sid, llm, hoodie = await _print_shop(client, monkeypatch, "09125550067")
    from app.services import chat_router
    seen: list[dict] = []

    async def router_json(*, task, system, prompt, **kw):
        seen.append({"task": task, "prompt": prompt, "system": system})
        return {"skill": next_skill.pop(0), "sure": True}
    next_skill: list[str] = []
    monkeypatch.setattr(chat_router.ai, "generate_json", router_json)
    monkeypatch.setattr(chat_router, "route", REAL_ROUTE)

    async def say(t, skill):
        next_skill.append(skill)
        return (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": t})).json()["messages"]

    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": "سلام"})
    n = len(llm.calls)
    last = (await say("دیگه چه مدل هودی دارین؟", "browse"))[-1]
    assert last["source"] == "catalog" and "هودی چاپ طرح دلخواه" in last["text"] and len(llm.calls) == n
    assert seen[-1]["task"] == "chat_router" and "<customer>دیگه چه مدل هودی دارین؟</customer>" in seen[-1]["prompt"]

    # a product question goes to the sales model, with the part named
    llm.answers = [{"reply": "قیمتش ۹۰۰ هزار تومنه 🌸", "intent": "product", "products": [hoodie["id"]]}]
    await say("هودی چاپی چنده؟", "product")
    assert "ROUTER: this message is about «product»" in llm.calls[-1]["prompt"]

    # "finish it" with a cart: the journey's next step, no model
    llm.answers = [{"reply": "باشه", "intent": "order", "order_action": "update",
                    "order": {"items": [{"product_id": hoodie["id"], "qty": 1, "variant": "سفید M"}]}}]
    await say("هودی چاپی سفید M میخوام", "cart")
    n = len(llm.calls)
    last = (await say("تمومه ثبتش کن", "checkout"))[-1]
    assert "محل چاپ" in last["text"] and last["buttons"] == [["جلو", "پشت"]] and len(llm.calls) == n

    # "off_topic" from the router while a phone number is in the message: the sales model reads it
    llm.answers = [{"reply": "شماره ثبت شد 🌸", "intent": "order"}]
    await say("09171234567", "off_topic")
    assert len(llm.calls) == n + 1

    # a person: handed over without the model, the cart kept
    last = (await say("آدمین رو میخوام", "human"))[-1]
    assert last["source"] == "handoff" and (await _state(sid)).get("order")
