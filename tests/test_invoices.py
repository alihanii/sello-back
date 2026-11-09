import io

from tests.conftest import login

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")


async def test_invoice_link_receipt_and_orders(client):
    h = await login(client, "09128810001")
    sid = (await client.post("/stores", json={"name": "فاکتوری"}, headers=h)).json()["id"]
    p = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "هودی چاپی", "price": 450000, "stock_qty": 5})).json()

    s = (await client.get(f"/stores/{sid}/invoice-settings", headers=h)).json()
    assert s["seller_name"] == "فاکتوری" and not s["ready"]
    r = await client.put(f"/stores/{sid}/invoice-settings", headers=h, json={"card_number": "6037-9911-1234-5678"})
    assert r.status_code == 200 and r.json()["card_number"] == "6037991112345678" and r.json()["ready"]
    assert (await client.put(f"/stores/{sid}/invoice-settings", headers=h, json={"card_number": "123"})).status_code == 422

    r = await client.post(f"/stores/{sid}/invoices", headers=h, json={
        "customer_name": "سارا", "items": [{"product_id": p["id"], "qty": 2}, {"title": "ارسال پیشتاز", "unit_price": 50000}],
        "discount_toman": 100000})
    assert r.status_code == 201, r.text
    inv = r.json()
    assert inv["number"] == 1001 and inv["total_toman"] == 2 * 450000 + 50000 - 100000
    assert inv["link"].endswith(f"/i/{inv['code']}")

    # Not an order yet: the customer hasn't paid.
    assert (await client.get(f"/stores/{sid}/orders", headers=h)).json()["total"] == 0

    pub = (await client.get(f"/public/invoices/{inv['code']}")).json()
    assert pub["payment"]["card_number"] == "6037991112345678" and pub["seller"]["seller_name"] == "فاکتوری"
    assert "product_id" not in pub["items"][0]
    assert (await client.get("/public/invoices/nope")).status_code == 404

    # A receipt image is required, plus phone and address.
    r = await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "123"})
    assert r.status_code == 422
    r = await client.post(f"/public/invoices/{inv['code']}/receipt",
                          data={"phone": "۰۹۱۲۱۱۱۲۲۳۳", "address": "تهران، ونک"},
                          files={"file": ("r.png", io.BytesIO(PNG), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "payment_submitted" and r.json()["receipt_sent"]
    assert r.json()["closed"] and r.json()["payment"]["card_number"] == ""
    # The link is closed now: no second receipt while the seller checks the first.
    again = await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "77"})
    assert again.status_code == 409

    orders = (await client.get(f"/stores/{sid}/orders", headers=h)).json()
    assert orders["total"] == 1 and orders["items"][0]["kind"] == "invoice"
    assert orders["items"][0]["customer_phone"] == "09121112233" and orders["counts"]["payment_submitted"] == 1
    notes = (await client.get("/notifications", headers=h)).json()
    assert any("1001" in n["title_fa"] for n in (notes["items"] if isinstance(notes, dict) else notes))

    r = await client.patch(f"/invoices/{inv['id']}", headers=h, json={"status": "confirmed"})
    assert r.status_code == 200 and r.json()["paid_at"]
    assert (await client.get(f"/products/{p['id']}", headers=h)).json()["stock_qty"] == 3
    r = await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "9"})
    assert r.status_code == 409

    # A chat order the seller logs by hand, already paid.
    r = await client.post(f"/stores/{sid}/invoices", headers=h, json={
        "source": "chat", "platform": "instagram", "customer_name": "علی", "status": "confirmed",
        "items": [{"title": "تیشرت", "unit_price": 200000, "qty": 1}]})
    assert r.status_code == 201, r.text
    assert r.json()["link"] == "" and r.json()["number"] == 1002

    orders = (await client.get(f"/stores/{sid}/orders?source=chat", headers=h)).json()
    assert orders["total"] == 1 and orders["sources"] == {"ai": 0, "invoice": 1, "chat": 1}

    f = (await client.get(f"/stores/{sid}/finance", headers=h)).json()
    assert f["by_source"]["invoice"] == {"toman": 850000, "orders": 1}
    assert f["by_source"]["chat"] == {"toman": 200000, "orders": 1}
    assert f["by_source"]["ai"]["toman"] == 0 and f["totals"]["income"] == 1050000

    d = (await client.get(f"/stores/{sid}/dashboard", headers=h)).json()
    assert d["kpis"]["income_today"] == 1050000 and d["kpis"]["all_orders_today"] == 2
    assert d["kpis"]["sales_total"] == 1050000

    # Other sellers can't see it.
    other = await login(client, "09128810002")
    assert (await client.get(f"/invoices/{inv['id']}", headers=other)).status_code == 404


async def test_rejected_receipt_reopens_and_cancelled_or_expired_closes(client):
    h = await login(client, "09128810003")
    sid = (await client.post("/stores", json={"name": "بسته"}, headers=h)).json()["id"]
    await client.put(f"/stores/{sid}/invoice-settings", headers=h, json={"card_number": "6037991112345678", "require_receipt": False,
                                                                         "ask_address": False, "ask_phone": False})
    mk = lambda: client.post(f"/stores/{sid}/invoices", headers=h, json={"items": [{"title": "x", "unit_price": 1000}]})
    inv = (await mk()).json()
    assert (await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "1"})).status_code == 200
    await client.patch(f"/invoices/{inv['id']}", headers=h, json={"status": "rejected"})
    pub = (await client.get(f"/public/invoices/{inv['code']}")).json()
    assert not pub["closed"] and pub["payment"]["card_number"]
    assert (await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "2"})).status_code == 200

    inv = (await mk()).json()
    await client.patch(f"/invoices/{inv['id']}", headers=h, json={"status": "cancelled"})
    pub = (await client.get(f"/public/invoices/{inv['code']}")).json()
    assert pub["closed"] and pub["payment"]["card_number"] == ""
    assert (await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "3"})).status_code == 409

    from datetime import timedelta
    from app.core.db import SessionLocal, utcnow
    from app.models import Invoice
    inv = (await mk()).json()
    async with SessionLocal() as db:
        row = await db.get(Invoice, inv["id"])
        row.expires_at = utcnow() - timedelta(hours=1)
        await db.commit()
    pub = (await client.get(f"/public/invoices/{inv['code']}")).json()
    assert pub["expired"] and pub["closed"] and pub["payment"]["card_number"] == ""
    assert (await client.post(f"/public/invoices/{inv['code']}/receipt", data={"payment_ref": "4"})).status_code == 409

    cal = (await client.get(f"/stores/{sid}/calendar?start=2020-01-01&end=2030-01-01", headers=h)).json()
    assert sum(v["own_orders"] for v in cal["days"].values()) == 1
