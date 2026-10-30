from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from tests.conftest import login


class FakeResp:
    def __init__(self, data):
        import json
        self.content = json.dumps(data).encode()


async def test_zarinpal_payment_activates_plan(client, monkeypatch):
    admin = await login(client, "09120000000")
    h = await login(client, "09127777777")
    r = await client.post("/payments/start", headers=h, json={"plan_id": 1})
    assert r.status_code == 400  # disabled until the admin configures a gateway

    r = await client.put("/admin/integrations/payment", headers=admin, json={
        "enabled": True, "bank": "ZARINPAL", "gateways": {"ZARINPAL": {"MERCHANT_CODE": "abc-123", "SANDBOX": True}}})
    assert r.status_code == 200, r.text
    assert r.json()["gateways"]["ZARINPAL"]["MERCHANT_CODE"] == "••••-123"

    sent = []

    def fake_post(url, json=None, timeout=None):
        sent.append((url, json))
        if url.endswith("request.json"):
            return FakeResp({"data": {"authority": "A0000000000000000000000000000123456", "code": 100, "message": "Success"}, "errors": []})
        return FakeResp({"data": {"code": 100, "message": "Verified", "ref_id": 987}, "errors": []})

    from app.services import bank
    import asyncio
    await asyncio.to_thread(bank._setup)
    import azbankgateways.banks.zarinpal as zp
    monkeypatch.setattr(zp.requests, "post", fake_post)

    plan = (await client.get("/plans")).json()[0]
    r = await client.post("/payments/start", headers=h, json={"plan_id": plan["id"]})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["url"] == "https://sandbox.zarinpal.com/pg/StartPay/A0000000000000000000000000000123456"
    req = sent[0][1]
    assert req["merchant_id"] == "abc-123" and req["amount"] == plan["price_toman"]  # Toman → Zarinpal (IRT)
    cb = urlparse(req["callback_url"])
    assert cb.path == "/api/v1/payments/callback/" and parse_qs(cb.query)["bank_type"] == ["ZARINPAL"]

    r = await client.get(f"/payments/callback/?{cb.query}&Authority=A0000000000000000000000000000123456&Status=OK",
                         follow_redirects=False)
    assert r.status_code == 303 and "payment=success" in r.headers["location"]
    sub = (await client.get("/me/subscription", headers=h)).json()["subscription"]
    assert sub and sub["plan"]["id"] == plan["id"]
    # A repeated callback must not create a second subscription.
    r = await client.get(f"/payments/callback/?{cb.query}&Authority=A0000000000000000000000000000123456&Status=OK",
                         follow_redirects=False)
    assert "payment=success" in r.headers["location"] or "payment=failed" in r.headers["location"]
    pays = (await client.get("/admin/payments", headers=admin)).json()
    assert sum(1 for p in pays if p["status"] == "paid") == 1
