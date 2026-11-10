from tests.conftest import login
from tests.test_chat import _no_ai_keys_left, _setup  # noqa: F401 - the fixture clears the AI key after the test
from tests.test_invoices import PNG


async def test_chat_link_page_lands_in_store_chats(client, monkeypatch):
    h, sid, _llm, *_ = await _setup(client, monkeypatch, "09125550901")
    link = (await client.get(f"/stores/{sid}/chat-link", headers=h)).json()
    assert link["slug"] == f"shop{sid}" and link["enabled"] and link["url"].endswith(f"/c/shop{sid}")
    for bad in ("Ab", "www", "a_b", "-abc"):
        assert (await client.put(f"/stores/{sid}/chat-link", headers=h, json={"slug": bad})).status_code == 422, bad
    r = await client.put(f"/stores/{sid}/chat-link", headers=h, json={"slug": "boutique-chat", "greeting": "سلام عزیزم"})
    assert r.status_code == 200 and r.json()["url"].endswith("/c/boutique-chat")
    # Another store can't take it.
    h2 = await login(client, "09125550902")
    other = (await client.post("/stores", json={"name": "دومی"}, headers=h2)).json()["id"]
    assert (await client.put(f"/stores/{other}/chat-link", headers=h2, json={"slug": "boutique-chat"})).status_code == 409
    assert (await client.get(f"/stores/{other}/chat-link", headers=h)).status_code == 404
    assert (await client.get(f"/public/chat/shop{sid}")).status_code == 404

    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    page = (await client.get("/public/chat/Boutique-Chat")).json()
    assert page["name"] == "بوتیک چت" and page["greeting"] == "سلام عزیزم" and page["assistant_on"]
    assert page["phone_button"]

    # A name is required; the assistant answers like on Telegram (the first message gets the welcome).
    assert (await client.post("/public/chat/boutique-chat/messages", json={"name": " ", "text": "سلام"})).status_code == 422
    r = (await client.post("/public/chat/boutique-chat/messages", json={"name": "سارا", "text": "سلام"})).json()
    visitor = r["visitor"]
    assert len(visitor) >= 16 and [m["role"] for m in r["messages"]] == ["customer", "ai"]

    # A photo, like a receipt or a design.
    r = await client.post("/public/chat/boutique-chat/photo", data={"name": "سارا", "visitor": visitor},
                          files={"file": ("r.png", PNG, "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["visitor"] == visitor and r.json()["messages"][0]["image_url"]
    bad = await client.post("/public/chat/boutique-chat/photo", data={"name": "سارا", "visitor": visitor},
                            files={"file": ("a.txt", b"hi", "text/plain")})
    assert bad.status_code == 422

    convs = (await client.get(f"/stores/{sid}/chat/conversations?platform=link", headers=h)).json()
    items = convs["items"] if isinstance(convs, dict) else convs
    assert len(items) == 1 and items[0]["customer_name"] == "سارا" and items[0]["seller_unread"]

    # The seller answers from the panel; the page picks it up by polling.
    r = await client.post(f"/chat/conversations/{items[0]['id']}/messages", headers=h, json={"text": "در خدمتم"})
    assert r.status_code in (200, 201), r.text
    hist = (await client.get("/public/chat/boutique-chat/messages", params={"visitor": visitor})).json()
    assert hist["name"] == "سارا" and hist["messages"][-1]["role"] == "seller"
    last = hist["messages"][-1]["id"]
    assert (await client.get("/public/chat/boutique-chat/messages",
                             params={"visitor": visitor, "after": last})).json()["messages"] == []
    assert (await client.get("/public/chat/boutique-chat/messages", params={"visitor": "x" * 20})).json()["messages"] == []

    await client.put(f"/stores/{sid}/chat-link", headers=h, json={"enabled": False})
    assert (await client.get("/public/chat/boutique-chat")).status_code == 403
    assert (await client.post("/public/chat/boutique-chat/messages", json={"name": "a", "text": "hi"})).status_code == 403


async def test_chat_link_caps_ai_replies_per_hour(client, monkeypatch):
    from app.routers import chat_link

    h, sid, _llm, *_ = await _setup(client, monkeypatch, "09125550903")
    await client.get(f"/stores/{sid}/chat-link", headers=h)
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    monkeypatch.setattr(chat_link, "AI_REPLIES_PER_HOUR", 1)
    r = (await client.post(f"/public/chat/shop{sid}/messages", json={"name": "رضا", "text": "سلام"})).json()
    assert [m["role"] for m in r["messages"]] == ["customer", "ai"] and not r["notice"]
    r = (await client.post(f"/public/chat/shop{sid}/messages",
                           json={"name": "رضا", "visitor": r["visitor"], "text": "قیمت مانتو؟"})).json()
    assert [m["role"] for m in r["messages"]] == ["customer"] and r["notice"]


async def test_chat_link_stream_pushes_new_messages(client, monkeypatch):
    import asyncio
    import json

    from app.core.db import SessionLocal
    from app.models import ChatConversation
    from app.routers import chat_link
    from app.services import chat_runtime as rt
    from app.services import live

    h, sid, _llm, *_ = await _setup(client, monkeypatch, "09125550904")
    await client.get(f"/stores/{sid}/chat-link", headers=h)
    r = (await client.post(f"/public/chat/shop{sid}/messages", json={"name": "نگار", "text": "سلام"})).json()
    visitor, first = r["visitor"], r["messages"][0]["id"]
    assert (await client.get(f"/public/chat/shop{sid}/stream", params={"visitor": "x" * 20})).status_code == 404

    # A commit with a new message wakes whoever waits on the conversation.
    async with SessionLocal() as db:
        conv = (await db.execute(chat_link.select(ChatConversation).where(
            ChatConversation.external_chat_id == visitor))).scalar_one()
        with live.watch(conv.id) as woke:
            rt.add_message(db, conv, role="seller", text="سلام نگار جان")
            assert not woke.is_set()
            await db.commit()
            assert woke.is_set()

    # The stream sends what is new after `after`, pings, and lets go when idle.
    monkeypatch.setattr(chat_link, "STREAM_PING", 0.05)
    monkeypatch.setattr(chat_link, "STREAM_IDLE", 0.1)
    async with client.stream("GET", f"/public/chat/shop{sid}/stream", params={"visitor": visitor, "after": first}) as s:
        body = await asyncio.wait_for(s.aread(), 5)
    text = body.decode()
    sent = json.loads(text.split("data: ", 1)[1].split("\n\n", 1)[0])
    assert sent[-1]["role"] == "seller" and all(m["id"] > first for m in sent)
    assert "event: bye" in text
