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
