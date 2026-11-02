import json

from app.platforms.base import CrawledItem
from app.services import ai, categories, jobs
from tests.conftest import drain, login


def test_match_reuses_the_same_name_written_differently():
    shop = ["پوشاک زنانه", "ماگ"]
    assert categories.match("پوشاك  زنانه", shop) == "پوشاک زنانه"  # Arabic ک, double space
    assert categories.match("پوشاک‌زنانه", shop) == "پوشاک زنانه"  # half-space
    assert categories.match(" اکسسوری ", shop) == "اکسسوری"
    assert categories.match(None, shop) == ""


async def test_ai_fills_category_from_the_shop_list(client, monkeypatch):
    h = await login(client, "09127770001")
    sid = (await client.post("/stores", json={"name": "دسته‌ها"}, headers=h)).json()["id"]
    r = await client.post(f"/stores/{sid}/products", headers=h,
                          json={"title": "بشقاب", "category": " ماگ ", "price": 100})
    assert r.json()["category"] == "ماگ"

    prompts = []

    async def fake(*, task, system, prompt, images=None, store_id=None, **kw):
        prompts.append(json.loads(prompt))
        return {"title": "ماگ سرامیکی سفید", "description": "ماگ", "category_hint": "ماگ سرامیکی",
                "category": "ماگ", "attributes": {}}

    monkeypatch.setattr(ai, "generate_json", fake)
    op = (await client.post(f"/stores/{sid}/ai-products", headers=h, json={"note": "ماگ سفید"})).json()
    await drain()
    op = (await client.get(f"/operations/{op['id']}", headers=h)).json()
    assert op["status"] == "succeeded", op
    assert prompts[-1]["shop_categories"] == ["ماگ"]
    prod = (await client.get(f"/products/{op['result']['product_id']}", headers=h)).json()
    assert prod["category"] == "ماگ" and prod["category_hint"] == "ماگ سرامیکی"

    cats = (await client.get(f"/stores/{sid}/product-categories", headers=h)).json()
    assert cats == [{"name": "ماگ", "count": 2}]
    listed = (await client.get(f"/stores/{sid}/products", params={"category": "ماگ"}, headers=h)).json()
    assert listed["total"] == 2
    assert (await client.get(f"/stores/{sid}/products", params={"category": "-"}, headers=h)).json()["total"] == 0


async def test_crawl_ai_mode_fills_category(monkeypatch):
    async def fake(*, task, system, prompt, store_id=None, **kw):
        assert json.loads(prompt)["shop_categories"] == ["هودی"]
        return {"posts": [{"message_id": "1", "title": "هودی", "category": "هودي"},
                          {"message_id": "2", "title": "ماگ", "category": "ماگ"}]}

    async def on(task):
        return True

    monkeypatch.setattr(ai, "ai_available", on)
    monkeypatch.setattr(ai, "generate_json", fake)
    items = [CrawledItem(external_id="1", caption="هودی"), CrawledItem(external_id="2", caption="ماگ")]
    out, _ = await jobs._extract(items, "llm", 1, ["هودی"])
    assert out["1"]["category"] == "هودی" and out["2"]["category"] == "ماگ"


async def test_bulk_edit_selected_and_all_matching(client):
    h = await login(client, "09127770002")
    sid = (await client.post("/stores", json={"name": "گروهی"}, headers=h)).json()["id"]
    ids = []
    for i, price in enumerate([100_000, 249_000, 1_000_000]):
        r = await client.post(f"/stores/{sid}/products", headers=h,
                              json={"title": f"محصول {i}", "price": price, "stock_qty": 5,
                                    "attributes": {"رنگ": "مشکی"}})
        ids.append(r.json()["id"])

    async def bulk(**body):
        r = await client.post(f"/stores/{sid}/products/bulk", headers=h, json={"action": "edit", **body})
        assert r.status_code == 200, r.text
        return r.json()

    async def get(pid):
        return (await client.get(f"/products/{pid}", headers=h)).json()

    # +10% rounded to the thousand, on two selected products
    r = await bulk(product_ids=ids[:2], edit={"price": {"mode": "percent", "value": 10, "round_to": 1000}})
    assert r["changed"] == 2
    assert [(await get(i))["price"] for i in ids] == [110_000, 274_000, 1_000_000]

    # every product in the shop (empty filter): stock -7 never goes below zero, category + attribute set
    r = await bulk(filters={}, edit={"stock": {"mode": "add", "value": -7}, "category": "پوشاک",
                                     "attribute": {"name": "جنس", "value": "نخی"}})
    assert r["affected"] == 3
    p = await get(ids[2])
    assert p["stock_qty"] == 0 and p["category"] == "پوشاک" and p["attributes"] == {"رنگ": "مشکی", "جنس": "نخی"}

    # filter by category, remove an attribute, set an exact price
    await bulk(filters={"category": "پوشاک", "q": "محصول 1"},
               edit={"attribute": {"name": "رنگ", "remove": True}, "price": {"mode": "set", "value": 99000}})
    p = await get(ids[1])
    assert p["attributes"] == {"جنس": "نخی"} and p["price"] == 99000
    assert (await get(ids[0]))["attributes"] == {"رنگ": "مشکی", "جنس": "نخی"}

    # clearing the category, and a fixed-amount decrease
    await bulk(product_ids=[ids[0]], edit={"category": "", "price": {"mode": "amount", "value": -10000}})
    p = await get(ids[0])
    assert p["category"] == "" and p["price"] == 100_000

    # nothing chosen / another shop's products
    r = await client.post(f"/stores/{sid}/products/bulk", headers=h, json={"action": "edit", "product_ids": ids, "edit": {}})
    assert r.status_code == 422
    other = await login(client, "09127770003")
    osid = (await client.post("/stores", json={"name": "دیگری"}, headers=other)).json()["id"]
    r = await client.post(f"/stores/{osid}/products/bulk", headers=other,
                          json={"action": "edit", "product_ids": ids, "edit": {"category": "x"}})
    assert r.json()["affected"] == 0


async def test_chat_uses_category():
    from app.core.db import SessionLocal, init_db
    from app.models import ChatAgent, Product, Store, User
    from app.services import chat_engine

    await init_db()
    async with SessionLocal() as db:
        user = User(phone="09127770009", name="")
        db.add(user)
        await db.flush()
        store = Store(owner_id=user.id, name="چت دسته")
        db.add(store)
        await db.flush()
        # the title doesn't say «هودی»; only the category does
        sweat = Product(store_id=store.id, title="سویشرت کلاهدار طوسی", price=500000, stock_qty=3, category="هودی")
        mug = Product(store_id=store.id, title="ماگ سفید", price=90000, stock_qty=3, category="ماگ")
        db.add_all([sweat, mug])
        await db.commit()
        found = await chat_engine.search_products(db, store.id, "هودی دارین؟", [])
        assert found[0].id == sweat.id
        assert "دسته: هودی" in chat_engine.product_card(sweat)
        agent = ChatAgent(store_id=store.id, catalog_enabled=True, orders_enabled=False)
        db.add(agent)
        await db.flush()  # column defaults
        prompt = chat_engine.static_prompt(store, agent, [], await categories.category_names(db, store.id))
        assert "CATEGORIES" in prompt and "هودی" in prompt and "ماگ" in prompt


def test_catalog_overview_lists_categories():
    from app.models import Product
    from app.services.chat_engine import catalog_overview

    rows = [Product(id=i, title=t, price=1000, stock_qty=1, status="ready", category=c)
            for i, (t, c) in enumerate([("هودی طوسی", "هودی"), ("هودی مشکی", "هودی"), ("ماگ", "ماگ")], 1)]
    text = catalog_overview(rows, True)
    assert "دسته‌ها: هودی، ماگ" in text and "هودی طوسی" in text
