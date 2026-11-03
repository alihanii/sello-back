"""Sales chat assistant: connections, the no-model paths, the tool-calling agent, orders, takeover, usage."""

import json

import pytest
from sqlalchemy import delete

from app.platforms.botapi import BotApiAdapter
from app.services import ai, chat_agent, chat_engine
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


USAGE = {"model": "gemini-3.6-flash", "provider_id": 1, "input_tokens": 1200, "output_tokens": 60,
         "cached_tokens": 800, "cost_toman": 3, "latency_ms": 420}


class FakeLLM:
    """Scripted JSON model for generate_chat (the FAQ import still uses it)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.answers: list[dict] = []

    async def __call__(self, *, system, prompt, store_id, max_tokens=400, task="chat"):
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        out = self.answers.pop(0) if self.answers else {"reply": "بله موجوده", "intent": "product"}
        return out, dict(USAGE)


class FakeAgent:
    """Scripted tool-calling model. Each step is a text reply or a list of (tool, args) calls. Records what each step
    was sent, so tests can check the prompt, the tools and the tool results."""

    def __init__(self):
        self.calls: list[dict] = []
        self.steps: list = []

    async def __call__(self, *, system, messages, tools, store_id, max_tokens=700, task="chat", pin=None):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": [t["name"] for t in tools]})
        step = self.steps.pop(0) if self.steps else "بله موجوده 🌸"
        if isinstance(step, str):
            return {"role": "assistant", "text": step, "calls": []}, dict(USAGE)
        calls = [{"id": f"c{i}", "name": n, "args": a} for i, (n, a) in enumerate(step)]
        return {"role": "assistant", "text": "", "calls": calls}, dict(USAGE)

    def first(self, i=-1) -> str:
        """The context message (state, products, chat, customer) of call i."""
        return self.calls[i]["messages"][0]["text"]

    def results(self, i=-1) -> list[str]:
        """Tool results the model saw at call i."""
        return [r["content"] for m in self.calls[i]["messages"] if m["role"] == "tool" for r in m["results"]]


KNOWLEDGE = {"shipping": {"cost": "۵۰ هزار تومان با پست", "time": "۲ تا ۴ روز کاری", "areas": "همه‌ی ایران"},
             "returns": {"policy": "exchange", "days": "۷", "conditions": "برچسب جدا نشده باشد"}}


async def _setup(client, monkeypatch, phone="09125550001", ready=True):
    monkeypatch.setattr(BotApiAdapter, "_call", fake_bot_call)
    llm = FakeAgent()
    monkeypatch.setattr(ai, "chat_step", llm)
    monkeypatch.setattr(ai, "generate_chat", FakeLLM())
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
    if ready:  # the required checklist steps, so the assistant can be turned on
        r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"knowledge": KNOWLEDGE})
        assert r.status_code == 200, r.text
    return h, sid, llm, p1, p2


ORDERS_ON = {"enabled": True, "orders_enabled": True,
             "payment": {"card_number": "6037-1111", "holder": "علی", "require_receipt": True}}


async def say(client, h, sid, text):
    """One message in the seller's test chat; returns (the replies, the conversation state)."""
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"text": text})).json()
    return r["messages"][1:], r["state"]


async def test_full_chat_flow(client, monkeypatch):
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550003")
    scheduled: list[int] = []
    monkeypatch.setattr(rt, "schedule", lambda cid: scheduled.append(cid))
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        **ORDERS_ON, "welcome_message": "سلام به {store}!",
        "faqs": [{"q": "هزینه ارسال چقدره؟", "a": "ارسال ۷۰ هزار تومان است."}]})
    bot = (await client.post(f"/stores/{sid}/chat/bots", headers=h, json={
        "platform": "telegram", "mode": "bot", "bot_token": "444:flow-token"})).json()

    # 1. "/start": welcome only, no model; a repeated update is ignored
    await _customer(bot["id"], 1001, "/start", 1)
    conv_id = scheduled[-1]
    saved = await rt.answer(conv_id)
    assert [m.source for m in saved] == ["welcome"] and saved[0].text == "سلام به بوتیک چت!" and llm.calls == []
    scheduled.clear()
    await _customer(bot["id"], 1001, "/start", 1)
    assert scheduled == []

    # 2. FAQ: the seller's own answer, no model
    await _customer(bot["id"], 1001, "هزینه ارسال چقدره", 2)
    saved = await rt.answer(conv_id)
    assert saved[0].source == "faq" and "۷۰" in saved[0].text and llm.calls == []

    # 3. a burst of two messages -> one turn; the matching product is in the message before the model asks
    await _customer(bot["id"], 1001, "سلام مانتو لینن", 3)
    await _customer(bot["id"], 1001, "سایز ۴۰ دارید؟", 4)
    llm.steps = ["بله، مانتو لینن کرم سایز ۴۰ موجوده؛ ۱٬۲۵۰٬۰۰۰ تومان 🌸"]
    saved = await rt.answer(conv_id)
    assert len(llm.calls) == 1 and saved[0].source == "llm" and saved[0].input_tokens == 1200
    call = llm.calls[0]
    assert "مانتو لینن کرم" in llm.first() and "<customer>سلام مانتو لینن\nسایز ۴۰ دارید؟</customer>" in llm.first()
    assert "هزینه ارسال" in call["system"] and "6037" not in call["system"]  # FAQ in the cached prefix, no card
    assert {"search_products", "cart_set_item", "show_order_summary", "confirm_order"} <= set(call["tools"])

    # 4. off-topic: the shop's fixed answer
    await _customer(bot["id"], 1001, "یه شعر درباره پاییز بگو", 5)
    llm.steps = ["OFF_TOPIC"]
    saved = await rt.answer(conv_id)
    assert saved[0].source == "off_topic" and "بوتیک چت" in saved[0].text

    # 5. the order: tools keep the cart; the phone with a wrong length is refused with the reason
    await _customer(bot["id"], 1001, "۹ تا سایز ۴۰ میخوام، مریم احمدی ۰۹۱۲۱۲۳۴۵۶ تهران خیابان آزادی", 6)
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 9, "variant": "۴۰"}),
                  ("set_customer_info", {"name": "مریم احمدی", "phone": "۰۹۱۲۱۲۳۴۵۶", "address": "تهران خیابان آزادی"})],
                 "فقط ۳ تا موجوده و گذاشتمش 🌸 شماره‌ات ۱۰ رقمیه؛ لطفاً شماره ۱۱ رقمی رو بفرست."]
    saved = await rt.answer(conv_id)
    cart, info = llm.results()
    assert "Only 3 in stock" in cart and "×3 = 3750000" in cart
    assert "ERROR" in info and "10 digits" in info and "STILL NEEDED: phone" in info
    assert saved[0].source == "llm" and "۱۱ رقمی" in saved[0].text

    await _customer(bot["id"], 1001, "ببخشید ۰۹۱۲۱۲۳۴۵۶۷", 7)
    llm.steps = [[("set_customer_info", {"phone": "۰۹۱۲۱۲۳۴۵۶۷"}), ("show_order_summary", {})]]
    saved = await rt.answer(conv_id)
    assert saved[0].source == "order" and "خلاصهٔ سفارش" in saved[0].text and "۳٬۷۵۰٬۰۰۰" in saved[0].text
    assert "09121234567" in saved[0].text

    # "بله" places it in code, no model
    calls = len(llm.calls)
    await _customer(bot["id"], 1001, "بله", 8)
    saved = await rt.answer(conv_id)
    assert len(llm.calls) == calls and saved[0].source == "order" and "6037-1111" in saved[0].text
    order = (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()["items"][0]
    assert order["status"] == "awaiting_payment" and order["total_toman"] == 3750000
    assert order["customer_phone"] == "09121234567" and order["items"][0]["variant"] == "۴۰"

    # 6. the receipt: a reference number completes payment, no model
    await _customer(bot["id"], 1001, "کد پیگیری ۱۲۳۴۵۶۷۸", 9)
    saved = await rt.answer(conv_id)
    assert saved[0].source == "receipt" and len(llm.calls) == calls
    order = (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()["items"][0]
    assert order["status"] == "payment_submitted" and order["payment_ref"] == "12345678"
    assert any(m == "sendMessage" and d.get("chat_id") == "1001" for m, d in SENT)

    # the seller confirms: stock goes down
    r = (await client.patch(f"/ai-orders/{order['id']}", headers=h, json={"status": "confirmed"})).json()
    assert r["status"] == "confirmed"
    assert (await client.get(f"/products/{p1['id']}", headers=h)).json()["stock_qty"] == 0

    # 7. the history shows everything, with the order linked
    detail = (await client.get(f"/chat/conversations/{conv_id}", headers=h)).json()
    assert detail["orders"][0]["id"] == order["id"]
    assert {m["source"] for m in detail["messages"] if m["role"] == "ai"} >= {"welcome", "faq", "llm", "off_topic", "order", "receipt"}

    # 8. the seller takes over: the assistant pauses for that chat
    r = await client.post(f"/chat/conversations/{conv_id}/messages", headers=h, json={"text": "سلام، من فروشنده‌ام"})
    assert r.status_code == 200
    await _customer(bot["id"], 1001, "یه سؤال دیگه هم داشتم درباره شال", 10)
    assert await rt.answer(conv_id) == []
    await client.post(f"/chat/conversations/{conv_id}/resume", headers=h)
    llm.steps = ["شال نخی مشکی الان ناموجوده 🙏"]
    saved = await rt.answer(conv_id)
    assert saved and saved[0].source == "llm"

    # 9. usage for the seller and the admin: one AI message per model-answered turn
    u = (await client.get(f"/stores/{sid}/chat/usage", headers=h)).json()
    assert u["totals"]["llm_calls"] == 5 and u["totals"]["orders"] == 1
    admin = await login(client, "09120000000")
    au = (await client.get("/admin/chat/usage", headers=admin)).json()
    assert any(s["store_id"] == sid and s["llm_calls"] == 5 for s in au["stores"])


async def test_no_order_without_a_summary_and_a_yes(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550020")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json=ORDERS_ON)
    await say(client, h, sid, "سلام")
    # the model tries to place it straight away: refused, and nothing is created
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "۴۲"}), ("confirm_order", {})],
                 "حتماً 🌸 اسم، شماره و آدرست رو بفرست."]
    await say(client, h, sid, "مانتو لینن ۴۲ میخوام ثبتش کن")
    assert "hasn't seen a summary" in llm.results()[1]
    assert (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()["items"] == []
    # the model says "ثبت شد" without placing anything: the shop sends the real summary instead
    # (told once to use the tools, it says it again)
    llm.steps = [[("set_customer_info", {"name": "سارا", "phone": "09120000001", "address": "قم"})],
                 "سفارشت ثبت شد 🌸", "سفارشت ثبت شد 🌸"]
    replies, state = await say(client, h, sid, "سارا ۰۹۱۲۰۰۰۰۰۰۱ قم")
    assert "nothing is in the order yet" in llm.calls[-1]["messages"][-1]["text"]
    assert replies[-1]["source"] == "order" and "خلاصهٔ سفارش" in replies[-1]["text"] and state["awaiting"] == "confirm"
    # confirm_order in the same turn as the summary is refused too: the customer must answer first
    llm.steps = [[("show_order_summary", {}), ("confirm_order", {})]]
    replies, state = await say(client, h, sid, "یه بار دیگه خلاصه رو بفرست")
    assert "خلاصهٔ سفارش" in replies[-1]["text"]
    assert (await client.get(f"/stores/{sid}/ai-orders", headers=h)).json()["items"] == []
    # a yes in other words, read by the model: now it is placed
    llm.steps = [[("confirm_order", {})]]
    replies, state = await say(client, h, sid, "همین خوبه بفرستید لطفا سریع")
    assert "6037-1111" in replies[-1]["text"] and state["awaiting"] == "receipt"


async def test_changes_after_the_summary_need_a_new_summary(client, monkeypatch):
    """ali's runs: "the address is wrong", "make it 2", "size 44 instead" after the summary."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550021")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json=ORDERS_ON)
    await say(client, h, sid, "سلام")
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "سایز ۴۰"}),
                  ("set_customer_info", {"name": "رضا", "phone": "+98 912 333 4455", "address": "تهران، ونک"}),
                  ("show_order_summary", {})]]
    replies, state = await say(client, h, sid, "مانتو لینن ۴۰، رضا ۰۹۱۲۳۳۳۴۴۵۵ تهران ونک")
    assert "09123334455" in replies[-1]["text"]
    first_sig = state["summary_sig"]
    # a change drops the old summary; a bare "yes" can't place the old version
    llm.steps = [[("set_customer_info", {"address": "شیراز، خیابان زند"}),
                  ("cart_set_item", {"product_id": p1["id"], "qty": 2})], "باشه 🌸 آدرس و تعداد عوض شد؛ خلاصه رو بفرستم؟"]
    replies, state = await say(client, h, sid, "آدرس اشتباهه شیراز خیابان زند، دوتا هم میخوام")
    assert "awaiting" not in state and "summary_sig" not in state
    assert state["order"]["items"] == [{"product_id": p1["id"], "qty": 2, "variant": "سایز ۴۰"}]  # size kept
    calls = len(llm.calls)
    llm.steps = [[("show_order_summary", {})]]
    replies, state = await say(client, h, sid, "بله")  # "yes" with no summary pending goes to the model
    assert len(llm.calls) == calls + 1 and "۲٬۵۰۰٬۰۰۰" in replies[-1]["text"] and "شیراز" in replies[-1]["text"]
    assert state["summary_sig"] != first_sig
    replies, state = await say(client, h, sid, "آره")
    # test-chat orders are hidden from /ai-orders, so the state is what tells it was placed
    assert state["awaiting"] == "receipt" and state.get("order_id")


async def test_size_choice_and_stock_are_checked_by_the_tool(client, monkeypatch):
    h, sid, llm, p1, p2 = await _setup(client, monkeypatch, "09125550022")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json=ORDERS_ON)
    await say(client, h, sid, "سلام")
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 1}),
                  ("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "۵۰"}),
                  ("cart_set_item", {"product_id": p2["id"], "qty": 1}),
                  ("cart_set_item", {"product_id": 999999, "qty": 1})], "چه سایزی؟ ۳۸ تا ۴۴ داریم."]
    replies, state = await say(client, h, sid, "مانتو و شال رو میخوام")
    no_size, bad_size, out, missing = llm.results()
    assert "choose a size" in no_size and "۳۸ تا ۴۴" in no_size
    assert "«۵۰» is not one of them" in bad_size
    assert "out of stock" in out and "no such product" in missing
    assert not (state.get("order") or {}).get("items")


async def test_a_placed_orders_details_are_fixed_on_the_order(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550023")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json=ORDERS_ON)
    await say(client, h, sid, "سلام")
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "۳۸"}),
                  ("set_customer_info", {"name": "نیما", "phone": "09125551234", "address": "کرج"}),
                  ("show_order_summary", {})]]
    await say(client, h, sid, "مانتو ۳۸ برای نیما ۰۹۱۲۵۵۵۱۲۳۴ کرج")
    replies, state = await say(client, h, sid, "بله")
    oid = state["order_id"]
    llm.steps = [[("set_customer_info", {"phone": "09125559999"})], "درست شد 🌸 شماره جدید ثبت شد."]
    replies, state = await say(client, h, sid, "شماره رو اشتباه دادم ۰۹۱۲۵۵۵۹۹۹۹")
    assert f"Updated on placed order #{oid}" in llm.results()[0]
    from app.core.db import SessionLocal
    from app.models import AiOrder
    async with SessionLocal() as db:
        assert (await db.get(AiOrder, oid)).customer_phone == "09125559999"
    # a phone number is never taken as a payment reference
    assert state["awaiting"] == "receipt"
    # a second order carries the contact details over; nothing is asked twice
    llm.steps = [[("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "۴۴"})], "یکی دیگه هم اضافه شد 🌸"]
    await say(client, h, sid, "یه ۴۴ هم اضافه کن")
    assert "STILL NEEDED" not in llm.results()[0] and "READY" in llm.results()[0]


async def test_numbers_the_data_never_gave_are_not_sent(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550024")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    await say(client, h, sid, "سلام")
    llm.steps = ["قیمتش ۹۹۰٬۰۰۰ تومانه 🌸", "قیمتش ۱٬۲۵۰٬۰۰۰ تومانه 🌸"]
    replies, _ = await say(client, h, sid, "قیمت مانتو لینن چنده؟")
    assert replies[-1]["text"] == "قیمتش ۱٬۲۵۰٬۰۰۰ تومانه 🌸"
    assert "990000" in llm.calls[-1]["messages"][-1]["text"]  # the model was told which number was wrong
    llm.steps = ["۹۹۰ هزار تومان", "باز هم ۹۹۰٬۰۰۰"]
    replies, _ = await say(client, h, sid, "مطمئنی؟")
    assert replies[-1]["text"] == chat_engine.NO_ANSWER_TEXT


async def test_photos_status_handoff_and_stock_policy(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550025")
    from app.core.db import SessionLocal
    from app.models import ProductImage
    async with SessionLocal() as db:
        db.add(ProductImage(product_id=p1["id"], url="/media/p1.jpg", position=0))
        await db.commit()
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={
        **ORDERS_ON, "answer_policy": {"stock": "available", "rules": ["تخفیف نده"]}})
    await say(client, h, sid, "سلام")
    # photos go out before the words
    llm.steps = [[("send_product_photos", {"product_id": p1["id"]})], "این هم عکسش 🌸 چه سایزی؟"]
    replies, _ = await say(client, h, sid, "عکس مانتو رو بفرست")
    assert replies[0]["image_url"] == "/media/p1.jpg" and replies[-1]["text"] == "این هم عکسش 🌸 چه سایزی؟"
    # the seller's stock rule: counts never reach the model and are removed if written anyway
    llm.steps = ["بله، فقط ۳ تا ازش مونده 🌸"]
    replies, _ = await say(client, h, sid, "مانتو لینن موجوده؟")
    assert "موجودی 3" not in llm.first() and "| موجود |" in llm.first() and "1. تخفیف نده" in llm.calls[-1]["system"]
    assert replies[-1]["text"] == "بله، موجوده 🌸"
    # status: only this chat's orders, sent by the shop's own text
    # no order in this chat: the model is told so and asks for the tracking code itself
    llm.steps = [[("check_order_status", {})], "سفارشی از این چت ثبت نشده؛ کد پیگیریت رو می‌فرستی؟"]
    replies, _ = await say(client, h, sid, "سفارشم چی شد؟")
    assert replies[-1]["source"] == "llm" and "کد پیگیری" in replies[-1]["text"]
    assert "no orders" in llm.results()[0]
    # a complaint the model hands over; asking for a person needs no model
    llm.steps = [[("handoff_to_human", {"reason": "complaint", "summary": "پارگی"})]]
    replies, state = await say(client, h, sid, "مانتویی که فرستادین پاره بود")
    assert replies[-1]["source"] == "handoff"
    await client.post(f"/stores/{sid}/chat/test", headers=h, json={"reset": True})
    await say(client, h, sid, "سلام")
    calls = len(llm.calls)
    replies, _ = await say(client, h, sid, "می‌خوام با ادمین صحبت کنم")
    assert replies[-1]["source"] == "handoff" and len(llm.calls) == calls


async def test_orders_off_hands_the_buyer_to_the_admin(client, monkeypatch):
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550026")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    await say(client, h, sid, "سلام")
    llm.steps = [[("pass_order_to_admin", {"product_id": p1["id"], "variant": "۴۰", "qty": 1})]]
    replies, state = await say(client, h, sid, "مانتو ۴۰ رو میخوام")
    assert "cart_set_item" not in llm.calls[-1]["tools"] and "pass_order_to_admin" in llm.calls[-1]["tools"]
    assert replies[-1]["source"] == "handoff" and "همکارم" in replies[-1]["text"]


async def test_model_failures(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550027")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    await say(client, h, sid, "سلام")

    async def boom(**kw):
        raise ai.AIError("پاسخ هوش مصنوعی قابل خواندن نبود.", code="ai_bad_json")
    monkeypatch.setattr(ai, "chat_step", boom)
    replies, _ = await say(client, h, sid, "مانتو دارید؟")
    assert replies[-1]["source"] == "canned" and "یه بار دیگه" in replies[-1]["text"]
    replies, _ = await say(client, h, sid, "مانتو دارید؟")
    assert replies[-1]["text"].startswith(chat_engine.AI_UNAVAILABLE_TEXT)


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
    llm.steps = ["۱٬۲۵۰٬۰۰۰ تومان"]
    await rt.answer(conv_id)
    assert all(d.get("business_connection_id") == "bc9" for m, d in SENT if m == "sendMessage")
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
    u = (await client.get(f"/stores/{sid}/chat/usage", headers=h)).json()
    assert u["totals"]["llm_calls"] == 0  # tests are not counted
    r = (await client.post(f"/stores/{sid}/chat/test", headers=h, json={"reset": True})).json()
    assert r["messages"] == []

    monkeypatch.setattr(chat_agent, "chat_limit", lambda db, store: _async(1))
    monkeypatch.setattr(chat_agent, "month_ai_messages", lambda db, sid: _async(1))
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


class _R:
    def __init__(self, body):
        self.status_code, self._body, self.text = 200, body, json.dumps(body)

    def json(self):
        return self._body


@pytest.mark.parametrize("kind", ["gemini", "openai", "anthropic"])
async def test_tool_calls_on_each_wire_format(monkeypatch, kind):
    from app.core.security import encrypt_credentials
    from app.models import AiProvider
    import httpx
    prov = AiProvider(name="x", kind=kind, api_key_encrypted=encrypt_credentials({"api_key": "k"}),
                      base_url="https://x.test/v1" if kind == "openai" else "")
    model = {"gemini": "gemini-3.6-flash", "openai": "gpt-x", "anthropic": "claude-haiku-5-5"}[kind]
    first = {
        "gemini": {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_product", "args": {"product_id": 5}}, "thoughtSignature": "sig1"}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5}},
        "openai": {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "get_product", "arguments": '{"product_id": 5}'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        "anthropic": {"content": [{"type": "tool_use", "id": "t1", "name": "get_product", "input": {"product_id": 5}}],
                      "usage": {"input_tokens": 100, "output_tokens": 5}},
    }[kind]
    second = {
        "gemini": {"candidates": [{"content": {"parts": [{"text": "بله 🌸"}]}}], "usageMetadata": {}},
        "openai": {"choices": [{"message": {"content": "بله 🌸"}}], "usage": {}},
        "anthropic": {"content": [{"type": "text", "text": "بله 🌸"}], "usage": {}},
    }[kind]
    bodies, replies = [], [_R(first), _R(second)]

    async def fake_post(self, url, headers=None, json=None):
        bodies.append(__import__("copy").deepcopy(json))
        return replies.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    tools = [{"name": "get_product", "description": "d", "parameters": {"type": "object", "properties": {
        "product_id": {"type": "integer"}}, "required": ["product_id"]}}]
    msgs = [{"role": "user", "text": "سلام"}]
    msg, tin, tout = await ai._call_tools(prov, model, "sys", msgs, tools, 0.3, 700, {})
    assert msg["calls"] == [{"id": msg["calls"][0]["id"], "name": "get_product", "args": {"product_id": 5}}] and tin == 100
    msgs += [msg, {"role": "tool", "results": [{"id": msg["calls"][0]["id"], "name": "get_product", "content": "#5 ..."}]}]
    msg, *_ = await ai._call_tools(prov, model, "sys", msgs, tools, 0.3, 700, {})
    assert msg["text"] == "بله 🌸" and not msg["calls"]
    sent = bodies[1]
    if kind == "gemini":
        assert sent["contents"][1]["parts"][0]["thoughtSignature"] == "sig1"  # Gemini 3 needs its signature back
        assert sent["contents"][2]["parts"][0]["functionResponse"]["name"] == "get_product"
        assert sent["tools"][0]["functionDeclarations"][0]["name"] == "get_product"
    elif kind == "openai":
        assert sent["messages"][2]["tool_calls"][0]["id"] == "t1" and sent["messages"][3]["tool_call_id"] == "t1"
    else:
        assert sent["messages"][1]["content"][0]["type"] == "tool_use"
        assert sent["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "#5 ..."}
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


async def test_gemini_retries_a_broken_tool_call_with_more_room(monkeypatch):
    """Gemini 3 at its least thinking sometimes ends with MALFORMED_FUNCTION_CALL and no parts: asked once more."""
    from app.core.security import encrypt_credentials
    from app.models import AiProvider
    import httpx
    prov = AiProvider(name="x", kind="gemini", api_key_encrypted=encrypt_credentials({"api_key": "k"}), base_url="")
    bodies, replies = [], [
        _R({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "content": {}}], "usageMetadata": {}}),
        _R({"candidates": [{"content": {"parts": [{"functionCall": {"name": "show_order_summary", "args": {}}}]}}],
            "usageMetadata": {}})]

    async def fake_post(self, url, headers=None, json=None):
        bodies.append(__import__("copy").deepcopy(json))
        return replies.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    tools = [{"name": "show_order_summary", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    meta: dict = {}
    msg, *_ = await ai._call_tools(prov, "gemini-3.6-flash", "sys", [{"role": "user", "text": "تایید"}], tools, 0.3, 700, meta)
    assert msg["calls"][0]["name"] == "show_order_summary" and meta["empty_reason"] == "MALFORMED_FUNCTION_CALL"
    # Gemini 3 runs at 1.0 (lower makes it loop) with low thinking, and its thinking gets room of its own on top of
    # the answer (thoughts count against maxOutputTokens; without room the answer was cut to nothing)
    first, retry = bodies[0]["generationConfig"], bodies[1]["generationConfig"]
    assert first["temperature"] == 1.0 and first["thinkingConfig"] == {"thinkingLevel": "low"}
    assert first["maxOutputTokens"] == 700 + ai.GEMINI_THINKING_ROOM
    assert retry["maxOutputTokens"] > first["maxOutputTokens"]


async def test_an_empty_gemini_answer_says_why_in_the_test_chat(client, monkeypatch):
    """The test chat shows the model's finish reason, so the next empty answer can be traced."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550034")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    await say(client, h, sid, "سلام")

    async def empty(**kw):
        raise ai.AIError("مدل پاسخ خالی داد.", code="ai_empty", detail="MAX_TOKENS, then MALFORMED_FUNCTION_CALL")
    monkeypatch.setattr(ai, "chat_step", empty)
    replies, _ = await say(client, h, sid, "اره")
    assert "MAX_TOKENS, then MALFORMED_FUNCTION_CALL" in replies[-1]["text"]


def test_unknown_numbers():
    known = "#12 | مانتو | 1250000 | موجودی 3\nCART: TOTAL: 2500000\n<customer>09121234567</customer>"
    assert chat_agent.unknown_numbers("قیمت ۱٬۲۵۰٬۰۰۰ و جمع ۲,۵۰۰,۰۰۰ تومان", known) == set()
    assert chat_agent.unknown_numbers("۱۲۵۰ هزار تومان، شماره ۰۹۱۲۱۲۳۴۵۶۷", known) == set()
    assert chat_agent.unknown_numbers("قیمت ۹۹۰٬۰۰۰", known) == {"990000"}


async def test_agent_settings_and_validation(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, ready=False)
    a = (await client.get(f"/stores/{sid}/chat/agent", headers=h)).json()
    assert a["enabled"] is False and "بوتیک چت" in a["defaults"]["welcome_message"]
    assert a["knowledge"]["shipping"]["cost"] == "" and [s["key"] for s in a["knowledge_form"]][:2] == ["shipping", "returns"]
    # the required checklist steps gate turning it on: shipping and returns must be filled first
    ov = (await client.get(f"/stores/{sid}/chat/overview", headers=h)).json()
    assert ov["ready"] is False
    assert {s["key"]: s["done"] for s in ov["setup"] if s["required"]} == {"ai": True, "products": True, "policies": False}
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    assert r.status_code == 422 and "هزینهٔ ارسال" in r.json()["detail"]["message"] and "مرجوعی" in r.json()["detail"]["message"]
    assert (await client.get(f"/stores/{sid}/chat/agent", headers=h)).json()["enabled"] is False
    # unknown sections, fields and return policies are dropped; text is trimmed
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"knowledge": {
        "shipping": {"cost": "  ۵۰ هزار  ", "time": "۳ روز", "hack": "x"}, "returns": {"policy": "maybe"}, "evil": {"a": 1}}})
    k = r.json()["knowledge"]
    assert k["shipping"]["cost"] == "۵۰ هزار" and "hack" not in k["shipping"] and k["returns"]["policy"] == "" and "evil" not in k
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    assert r.status_code == 422 and "مرجوعی" in r.json()["detail"]["message"] and "ارسال" not in r.json()["detail"]["message"]
    # filled in the same request it is turned on in
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True, "knowledge": KNOWLEDGE})
    assert r.status_code == 200 and r.json()["enabled"] is True
    assert (await client.get(f"/stores/{sid}/chat/overview", headers=h)).json()["ready"] is True
    # an assistant already on stays on when a field is cleared; only turning it on is gated
    r = await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"knowledge": {}})
    assert r.status_code == 200 and r.json()["enabled"] is True
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


async def test_faq_extract_from_text_and_file(client, monkeypatch):
    h, sid, *_ = await _setup(client, monkeypatch, "09125550010")
    llm = FakeLLM()
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


async def test_shop_policies_reach_the_assistant(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550031")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True, "knowledge": {
        **KNOWLEDGE, "sizing": {"guide": "۳۸ برای قد ۱۶۰ تا ۱۶۵"}, "products": {"care": "با آب سرد بشویید"}}})
    await say(client, h, sid, "سلام")
    # the shop's own price for shipping, written differently, is a known number, not a made-up one
    llm.steps = ["هزینه‌ی ارسال ۵۰٬۰۰۰ تومانه و ۲ تا ۴ روز کاری می‌رسه 🌸"]
    replies, _ = await say(client, h, sid, "ارسال چقدر طول میکشه و هزینش چنده؟")
    system = llm.calls[-1]["system"]
    assert "SHOP POLICIES" in system and "ارسال / هزینهٔ ارسال: ۵۰ هزار تومان با پست" in system
    assert "مرجوعی و تعویض / مرجوعی: فقط تعویض دارد" in system and "شستشو و نگهداری: با آب سرد بشویید" in system
    assert len(llm.calls) == 1 and replies[-1]["text"].startswith("هزینه‌ی ارسال ۵۰٬۰۰۰")


async def test_the_model_skipping_the_tools_is_sent_back_to_them(client, monkeypatch):
    """ali's chat: the model chatted about the order and asked for «تایید» without ever calling a tool."""
    h, sid, llm, p1, _ = await _setup(client, monkeypatch, "09125550032")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json=ORDERS_ON)
    await say(client, h, sid, "سلام")
    llm.steps = ["سفارش شما آماده ثبت است. لطفا برای تایید نهایی، &#34;تایید&#34; را بفرستید.",
                 [("cart_set_item", {"product_id": p1["id"], "qty": 1, "variant": "۴۰"}),
                  ("set_customer_info", {"name": "هانی", "phone": "09131343141", "address": "قم خیابان سلام پلاک ۲۲"}),
                  ("show_order_summary", {})]]
    replies, state = await say(client, h, sid, "مانتو ۴۰ هانی ۰۹۱۳۱۳۴۳۱۴۱ قم خیابان سلام پلاک ۲۲")
    note = llm.calls[1]["messages"][-1]["text"]
    assert "nothing is in the order yet" in note and "CART: empty" in note
    assert replies[-1]["source"] == "order" and state["awaiting"] == "confirm"
    replies, state = await say(client, h, sid, "تایید")
    assert state["awaiting"] == "receipt" and state.get("order_id")


async def test_error_notes_stay_out_of_the_models_history(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550033")
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    await say(client, h, sid, "سلام")

    async def empty(**kw):
        raise ai.AIError("مدل پاسخ خالی داد.", code="ai_empty")
    monkeypatch.setattr(ai, "chat_step", empty)
    replies, _ = await say(client, h, sid, "هودی داری؟")
    assert "خطای هوش مصنوعی" in replies[-1]["text"]
    monkeypatch.setattr(ai, "chat_step", llm)
    llm.steps = ["آره داریم 🌸 &quot;مشکی&quot;"]
    replies, _ = await say(client, h, sid, "هودی داری؟")
    assert "خطای هوش مصنوعی" not in llm.first() and "نتونستم" not in llm.first()
    assert replies[-1]["text"] == 'آره داریم 🌸 "مشکی"'


async def test_unwritten_shop_policies_are_named_to_the_model(client, monkeypatch):
    """The test chat works while the assistant is off, before the forms are filled: the model is told what the shop
    hasn't written (it made up a shipping cost in ali's test), and that a few products are not all of them."""
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550035", ready=False)
    await say(client, h, sid, "سلام")
    llm.steps = ["این رو از همکارم می‌پرسم 🌸"]
    await say(client, h, sid, "هزینه ارسال چقدره؟")
    assert "NOT WRITTEN BY THE SHOP: هزینهٔ ارسال، زمان رسیدن سفارش، شرایط مرجوعی" in llm.first().replace(", ", "، ")
    assert "never say \"only\" from PRODUCTS alone" in llm.calls[-1]["system"]
