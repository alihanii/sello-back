import io

from tests.conftest import login
from tests.test_invoices import PNG


async def test_site_settings_catalog_checkout_and_sitemap(client):
    h = await login(client, "09128820001")
    sid = (await client.post("/stores", json={"name": "سایت‌دار"}, headers=h)).json()["id"]
    hoodie = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "هودی چاپی مشکی", "price": 450000, "compare_at_price": 520000, "stock_qty": 3, "category": "هودی",
        "attributes": {"سایز": "M، L، XL", "جنس": "دورس پنبه"}})).json()
    mug = (await client.post(f"/stores/{sid}/products", headers=h, json={
        "title": "ماگ سرامیکی", "price": 120000, "stock_qty": 0, "category": "ماگ"})).json()

    s = (await client.get(f"/stores/{sid}/site", headers=h)).json()
    assert not s["enabled"] and s["subdomain"] == "" and s["suggested_subdomain"] == f"shop{sid}"
    assert s["config"]["palette"]["primary"] == "#28594d" and not s["payment_ready"]

    for bad in ("Ab", "www", "a_b", "-abc"):
        assert (await client.put(f"/stores/{sid}/site", headers=h, json={"subdomain": bad})).status_code == 422, bad
    assert (await client.put(f"/stores/{sid}/site", headers=h, json={"palette": {"primary": "red"}})).status_code == 422
    enamad = ("<a referrerpolicy='origin' target='_blank' href='https://trustseal.enamad.ir/?id=123&Code=abc'>"
              "<img referrerpolicy='origin' src='https://trustseal.enamad.ir/logo.aspx?id=123&Code=abc' alt='' "
              "style='cursor:pointer' code='abc'></a><script>alert(1)</script>")
    r = await client.put(f"/stores/{sid}/site", headers=h, json={
        "enabled": True, "subdomain": "hoodiland", "tagline": "هودی با طرح خودت",
        "palette": {"primary": "#1E3A8A", "accent": "#F59E0B"},
        "banners": [{"image_url": "/media/1/b.jpg", "title": "حراج پاییزه", "link": "/products?sale=1"}],
        "about": {"text": "ما از ۱۴۰۰ هودی می‌دوزیم."}, "contact": {"phone": "۰۹۱۲۰۰۰۱۱۲۲", "instagram": "hoodiland"},
        "badges": [{"kind": "enamad", "code": enamad}], "shipping": {"fee_toman": 50000, "free_over_toman": 1000000},
        "seo": {"description": "خرید هودی چاپی"}})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["enabled"] and s["subdomain"] == "hoodiland" and s["config"]["palette"]["primary"] == "#1e3a8a"
    assert s["config"]["contact"]["phone"] == "09120001122"
    assert s["config"]["badges"] == [{"kind": "enamad", "title": "اینماد", "href": "https://trustseal.enamad.ir/?id=123&Code=abc",
                                      "img": "https://trustseal.enamad.ir/logo.aspx?id=123&Code=abc"}]
    bad_badge = {"badges": [{"kind": "custom", "code": "<a href='javascript:alert(1)'>x</a>"}]}
    assert (await client.put(f"/stores/{sid}/site", headers=h, json=bad_badge)).status_code == 422

    # Someone else can't take the subdomain or the domain.
    h2 = await login(client, "09128820002")
    sid2 = (await client.post("/stores", json={"name": "دیگری"}, headers=h2)).json()["id"]
    assert (await client.put(f"/stores/{sid2}/site", headers=h2, json={"subdomain": "hoodiland"})).status_code == 409
    r = await client.put(f"/stores/{sid}/site", headers=h, json={"custom_domain": "https://www.Hoodiland.ir/"})
    assert r.json()["custom_domain"] == "hoodiland.ir" and r.json()["domain_status"] == "pending"
    assert (await client.put(f"/stores/{sid2}/site", headers=h2, json={"custom_domain": "hoodiland.ir"})).status_code == 409

    # The storefront finds the site by host: our subdomain (*.localhost in dev) or the own domain.
    for host in ("hoodiland.localhost:3000", "hoodiland.ir", "www.hoodiland.ir"):
        r = await client.get("/public/sites/resolve", params={"host": host})
        assert r.status_code == 200, host
    site = r.json()
    assert site["key"] == "hoodiland" and site["name"] == "سایت‌دار"
    assert site["categories"] == [{"name": "هودی", "count": 1, "path": "/category/%D9%87%D9%88%D8%AF%DB%8C"},
                                  {"name": "ماگ", "count": 1, "path": "/category/%D9%85%D8%A7%DA%AF"}] or \
        {c["name"] for c in site["categories"]} == {"هودی", "ماگ"}
    assert site["config"]["banners"][0]["image_url"].endswith("/media/1/b.jpg")
    assert site["config"]["banners"][0]["image_url"].startswith("http")
    assert not site["checkout_ready"]
    assert (await client.get("/public/sites/resolve", params={"host": "nope.localhost"})).status_code == 404
    assert (await client.get("/public/sites/tls-check", params={"domain": "hoodiland.ir"})).status_code == 200
    assert (await client.get("/public/sites/tls-check", params={"domain": "evil.com"})).status_code == 404

    lst = (await client.get("/public/sites/hoodiland/products")).json()
    assert lst["total"] == 2 and lst["items"][0]["id"] == hoodie["id"]  # in stock first
    assert lst["items"][0]["compare_at_price"] == 520000 and lst["items"][0]["few_left"]
    assert (await client.get("/public/sites/hoodiland/products?category=ماگ")).json()["total"] == 1
    assert (await client.get("/public/sites/hoodiland/products?available=true")).json()["total"] == 1
    assert (await client.get("/public/sites/hoodiland/products?q=ماگ")).json()["items"][0]["id"] == mug["id"]

    p = (await client.get(f"/public/sites/hoodiland/products/{hoodie['id']}")).json()
    assert p["options"] == [{"key": "size", "label": "سایز", "values": ["M", "L", "XL"]}]
    assert p["specs"] == [{"name": "جنس", "value": "دورس پنبه"}]
    assert p["path"].startswith(f"/product/{hoodie['id']}/") and p["in_stock"] and p["stock_qty"] == 3
    assert (await client.get("/public/sites/hoodiland/products/999999")).status_code == 404

    # Drafts and other stores' products are not on the site.
    other = (await client.post(f"/stores/{sid2}/products", headers=h2, json={"title": "x", "price": 1})).json()
    assert (await client.get(f"/public/sites/hoodiland/products/{other['id']}")).status_code == 404

    q = (await client.post("/public/sites/hoodiland/cart/quote", json={"items": [
        {"product_id": hoodie["id"], "qty": 5, "variant": "L"}, {"product_id": mug["id"], "qty": 1}]})).json()
    assert q["items"][0]["available_qty"] == 3 and len(q["problems"]) == 2
    assert q["subtotal_toman"] == 3 * 450000 and q["shipping_toman"] == 0  # over the free-shipping line

    order = {"items": [{"product_id": hoodie["id"], "qty": 1, "variant": "L"}], "name": "مریم",
             "phone": "۰۹۱۲۳۳۳۴۴۵۵", "address": "تهران، خیابان آزادی، پلاک ۱۲"}
    r = await client.post("/public/sites/hoodiland/checkout", json=order)
    assert r.status_code == 409  # no card number or payment link yet
    await client.put(f"/stores/{sid}/invoice-settings", headers=h, json={"card_number": "6037991112345678"})
    assert (await client.post("/public/sites/hoodiland/checkout", json={**order, "phone": "123"})).status_code == 422
    r = await client.post("/public/sites/hoodiland/checkout", json={**order, "items": [{"product_id": hoodie["id"]}]})
    assert r.status_code == 422  # size not chosen
    r = await client.post("/public/sites/hoodiland/checkout", json=order)
    assert r.status_code == 201, r.text
    placed = r.json()
    assert placed["total_toman"] == 450000 + 50000

    # It is in the orders list at once, marked as from the site.
    orders = (await client.get(f"/stores/{sid}/orders", headers=h)).json()
    assert orders["total"] == 1 and orders["items"][0]["kind"] == "site"
    assert orders["items"][0]["customer_phone"] == "09123334455" and orders["sources"]["site"] == 1
    assert (await client.get(f"/stores/{sid}/orders?source=site", headers=h)).json()["total"] == 1

    # The customer pays on the order page (the public invoice API) and the seller confirms.
    pub = (await client.get(f"/public/invoices/{placed['code']}")).json()
    assert pub["source"] == "site" and pub["payment"]["card_number"] == "6037991112345678"
    r = await client.post(f"/public/invoices/{placed['code']}/receipt",
                          files={"file": ("r.png", io.BytesIO(PNG), "image/png")})
    assert r.status_code == 200, r.text
    inv_id = orders["items"][0]["id"]
    assert (await client.patch(f"/invoices/{inv_id}", headers=h, json={"status": "confirmed"})).status_code == 200
    f = (await client.get(f"/stores/{sid}/finance", headers=h)).json()
    assert f["by_source"]["site"] == {"toman": 500000, "orders": 1} and f["by_source"]["ai"]["toman"] == 0
    assert (await client.get(f"/products/{hoodie['id']}", headers=h)).json()["stock_qty"] == 2

    xml = (await client.get("/public/sites/hoodiland/sitemap.xml", params={"base": "https://hoodiland.ir"})).text
    assert xml.startswith("<?xml") and "<loc>https://hoodiland.ir/</loc>" in xml
    assert f"<loc>https://hoodiland.ir/product/{hoodie['id']}/" in xml and "/category/" in xml
    assert "<image:image>" not in xml  # no pictures on these products

    # Turned off: gone from the web.
    await client.put(f"/stores/{sid}/site", headers=h, json={"enabled": False})
    assert (await client.get("/public/sites/hoodiland")).status_code == 404


async def test_site_chat_widget_lands_in_store_chats(client):
    h = await login(client, "09128820003")
    sid = (await client.post("/stores", json={"name": "چتی"}, headers=h)).json()["id"]
    await client.put(f"/stores/{sid}/site", headers=h, json={"enabled": True, "subdomain": "chatty"})
    site = (await client.get("/public/sites/chatty")).json()
    assert site["config"]["chat"]["enabled"] is False  # the assistant is off, so no widget

    r = await client.post("/public/sites/chatty/chat", json={"text": "سلام قیمت هودی؟", "page": "/product/1"})
    assert r.status_code == 200, r.text
    visitor = r.json()["visitor"]
    assert len(visitor) >= 16 and r.json()["messages"][0]["role"] == "customer"

    convs = (await client.get(f"/stores/{sid}/chat/conversations?platform=site", headers=h)).json()
    items = convs["items"] if isinstance(convs, dict) else convs
    assert len(items) == 1 and items[0]["platform"] == "site" and items[0]["seller_unread"]

    # The seller answers from the panel; the widget picks it up by polling.
    r = await client.post(f"/chat/conversations/{items[0]['id']}/messages", headers=h, json={"text": "سلام، ۴۵۰ تومن"})
    assert r.status_code in (200, 201), r.text
    hist = (await client.get("/public/sites/chatty/chat", params={"visitor": visitor, "after": 0})).json()
    assert [m["role"] for m in hist["messages"]] == ["customer", "seller"]
    last = hist["messages"][-1]["id"]
    assert (await client.get("/public/sites/chatty/chat", params={"visitor": visitor, "after": last})).json()["messages"] == []
    assert (await client.get("/public/sites/chatty/chat", params={"visitor": "x" * 20})).json()["messages"] == []

    await client.put(f"/stores/{sid}/site", headers=h, json={"chat": {"enabled": False}})
    assert (await client.post("/public/sites/chatty/chat", json={"text": "hi"})).status_code == 403
