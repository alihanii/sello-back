"""Digikala after publishing: admin switch, review state → price + stock, orders into the unified list."""

from app.core.db import SessionLocal
from app.core.security import encrypt_credentials
from app.models import Channel, ChannelListing, Product
from app.platforms import digikala as dk
from app.services import dk_sync
from tests.conftest import login

ADMIN = "09120000000"


def test_normalize_order_reads_common_shapes():
    o = dk.normalize_order({"id": 9, "order_code": "DK-1", "status": {"code": "shipped", "title": "ارسال شده"},
                            "created_at": "2026-11-13 10:00:00", "customer": {"full_name": "رضا", "city": "تهران"},
                            "items": [{"product_id": 55, "variant_id": 7, "title_fa": "هودی", "quantity": 2,
                                       "selling_price": 4500000}]})
    assert o["id"] == "9" and o["code"] == "DK-1" and o["status"] == "confirmed" and o["status_fa"] == "ارسال شده"
    assert o["customer_name"] == "رضا" and o["city"] == "تهران"
    assert o["items"][0] == {"remote_product_id": 55, "variant_id": 7, "title": "هودی", "qty": 2, "unit_price": 450000,
                             "image_url": ""}
    assert dk.normalize_order({"id": 1, "status": "cancelled_by_customer"})["status"] == "cancelled"
    assert dk.review_state("waiting_for_confirm") == "in_review"
    assert dk.review_state("rejected") == "rejected" and dk.review_state("need_to_edit") == "needs_edit"
    assert dk.review_state("approved") == "approved"


async def test_admin_switch_and_sync(client, monkeypatch):
    admin = await login(client, ADMIN)
    h = await login(client, "09128830771")
    sid = (await client.post("/stores", json={"name": "دیجی"}, headers=h)).json()["id"]
    p = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "هودی چاپی مدل A", "price": 450000, "stock_qty": 5})).json()

    # Admin switch: off blocks connecting and the platform card says why.
    r = await client.put("/admin/integrations/digikala", headers=admin, json={"enabled": False, "sync_minutes": 10})
    assert r.status_code == 200 and r.json()["enabled"] is False and r.json()["sync_minutes"] == 10
    row = next(x for x in (await client.get("/platforms")).json() if x["platform"] == "digikala")
    assert row["enabled"] is False and row["reason"] and row["can_crawl"] is False
    r = await client.post(f"/stores/{sid}/channels/digikala/send-code", headers=h, json={"platform": "digikala", "data": {"phone": "09121234567"}})
    assert r.status_code == 400
    assert (await client.put("/admin/integrations/digikala", headers=admin, json={"enabled": True})).json()["enabled"]

    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="digikala", status="connected", display_name="دیجی",
                     credentials_encrypted=encrypt_credentials({"access_token": "tok"}), config={})
        db.add(ch)
        await db.flush()
        db.add(ChannelListing(product_id=p["id"], channel_id=ch.id, status="published", remote_id="DKP-123",
                              remote_status="in_review"))
        await db.commit()
        cid = ch.id

    # Crawl from Digikala is gone.
    r = await client.post(f"/stores/{sid}/crawl-sources", headers=h, json={"type": "marketplace_listing", "channel_id": cid})
    assert r.status_code == 422

    calls = []
    orders = [{"id": 501, "order_code": "DK-501", "status": {"code": "new", "title": "جدید"}, "created_at": "2026-11-13T09:00:00",
               "customer": {"full_name": "مریم", "city": "شیراز"},
               "items": [{"product_id": 123, "variant_id": 9, "title_fa": "هودی", "quantity": 1, "selling_price": 4500000}]}]

    async def fake_request(self, method, path, *, token=None, body=None, referer=None, files=None, data=None):
        calls.append((method, path.split("?")[0], body))
        if path.startswith("/api/v2/products/filters/seller"):
            return {"data": [[{"name": "moderation_status", "data": {"approved": "تایید شده"}}]]}
        if path.startswith("/api/v2/products/seller"):
            return {"data": {"items": [{"product_id": 123, "status_data": "approved", "variants_count": 0}],
                             "pager": {"total_pages": 1}}}
        if path.startswith("/api/v2/variant-creation/v2/"):
            return {"data": {"warranties": [{"id": 3, "title": "ضمانت"}], "colors": [{"id": 1, "title": "مشکی"}]}}
        if path.startswith("/api/v2/variant-creation/variants/"):
            return {"data": {"variant_id": 777}}
        if path.startswith("/api/v2/ship-by-seller-orders"):
            return {"data": {"items": orders, "pager": {"total_pages": 1}}}
        if path.startswith("/api/v2/orders/"):
            raise dk.PlatformError("نه", "http_404")
        return {"data": {}}

    monkeypatch.setattr(dk.DigikalaAdapter, "_request", fake_request)

    # First sync: the product is approved → variant with price (rial) and stock; the old order is history only.
    r = await client.post(f"/channels/{cid}/digikala/sync", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["listings"]["live"] == 1 and r.json()["orders"]["new"] == 1
    variant = next(b for m, path, b in calls if path == "/api/v2/variant-creation/variants/")
    assert variant["product_id"] == 123 and variant["price"] == 4500000 and variant["seller_stock"] == 5
    assert variant["warranty_id"] == 3 and variant["theme_value_id"] == 1

    prod = (await client.get(f"/products/{p['id']}", headers=h)).json()
    ln = prod["listings"][0]
    assert ln["remote_status"] == "live"

    listed = (await client.get(f"/stores/{sid}/orders?source=digikala", headers=h)).json()
    assert listed["total"] == 1 and listed["sources"]["digikala"] == 1
    item = listed["items"][0]
    assert item["kind"] == "digikala" and item["ref"] == "DK-501" and item["external_status_fa"] == "جدید"
    assert item["items"][0]["product_id"] == p["id"] and item["total_toman"] == 450000
    async with SessionLocal() as db:
        assert (await db.get(Product, p["id"])).stock_qty == 5  # history from before the first sync takes no stock

    # Next sync: the order shipped (status follows), a new order arrives and takes stock.
    orders[0]["status"] = {"code": "shipped", "title": "ارسال شده"}
    orders.append({**orders[0], "id": 502, "order_code": "DK-502", "status": {"code": "new", "title": "جدید"}})
    r = await client.post(f"/channels/{cid}/digikala/sync", headers=h)
    assert r.json()["orders"] == {"orders": 2, "new": 1, "updated": 1}
    listed = (await client.get(f"/stores/{sid}/orders?source=digikala", headers=h)).json()
    assert {i["external_status_fa"] for i in listed["items"]} == {"ارسال شده", "جدید"}
    async with SessionLocal() as db:
        assert (await db.get(Product, p["id"])).stock_qty == 4
    # The seller can't change a Digikala order's status here.
    r = await client.patch(f"/invoices/{listed['items'][0]['id']}", headers=h, json={"status": "cancelled"})
    assert r.status_code == 409
    assert dk_sync  # module imported for the poller


async def test_send_code_runs_inline_and_saves_the_challenge(client, monkeypatch):
    h = await login(client, "09128830998")
    sid = (await client.post("/stores", json={"name": "کد"}, headers=h)).json()["id"]
    sent = []

    async def fake(self, method, path, *, token=None, body=None, referer=None, files=None, data=None):
        sent.append((path, body))
        return {"data": {"challenge_token": "ct"}} if path.endswith("/otp/send") else {"data": {}}

    monkeypatch.setattr(dk.DigikalaAdapter, "_request", fake)
    r = await client.post(f"/stores/{sid}/channels/digikala/send-code", headers=h,
                          json={"platform": "digikala", "data": {"phone": "09121234567"}})
    assert r.status_code == 200, r.text
    assert [p for p, _ in sent] == ["/api/v2/login", "/api/v2/otp/send"] and sent[0][1]["username"] == "09121234567"
    async with SessionLocal() as db:
        ch = await db.get(Channel, r.json()["channel_id"])
        assert ch.config["pending"] == {"challenge_token": "ct", "phone": "09121234567"}

    async def refuse(self, method, path, **kw):
        raise dk.PlatformError("کپچا لازم است", "http_400")

    monkeypatch.setattr(dk.DigikalaAdapter, "_request", refuse)
    r = await client.post(f"/stores/{sid}/channels/digikala/send-code", headers=h,
                          json={"platform": "digikala", "data": {"phone": "09121234567"}})
    assert r.status_code == 400 and r.json()["detail"]["message"] == "کپچا لازم است"
