"""Plans from the admin panel: edit, add, stop selling, delete; change a seller's plan; block; no AI after a plan ends."""

from datetime import timedelta

from app.core.db import SessionLocal, utcnow
from app.models import AiUsageLog, Subscription, User
from app.services import chat_engine, plans
from app.services import chat_runtime as rt
from tests.conftest import login
from tests.test_chat import _real_conv, _setup


async def _user_id(phone: str) -> int:
    from sqlalchemy import select
    async with SessionLocal() as db:
        return (await db.execute(select(User.id).where(User.phone == phone))).scalar_one()


async def test_admin_manages_plans(client):
    admin = await login(client, "09120000000")
    tiers = (await client.get("/admin/plan-tiers", headers=admin)).json()
    assert [t["tier"] for t in tiers] == ["sello"] and len(tiers[0]["prices"]) == 3
    assert tiers[0]["limits"]["monthly_chat_messages"] == 2000

    # a second plan, then stop selling it, then bring it back
    new = {**tiers[0], "tier": "", "name": "ویژه", "sort_order": 5, "is_featured": True,
           "limits": {**tiers[0]["limits"], "monthly_chat_messages": 9000}}
    r = await client.put("/admin/plan-tiers/vip", headers=admin, json=new)
    assert r.status_code == 200 and r.json()["is_featured"] and r.json()["prices"][0]["plan_id"]
    public = (await client.get("/plans")).json()
    assert {p["tier"] for p in public} == {"sello", "vip"}
    assert not any(p["is_featured"] for p in public if p["tier"] == "sello")  # one highlighted plan
    r = await client.post("/admin/plan-tiers/vip/active", headers=admin, json={"is_active": False})
    assert r.status_code == 200 and r.json()["is_active"] is False
    assert {p["tier"] for p in (await client.get("/plans")).json()} == {"sello"}
    assert (await client.post("/admin/plan-tiers/vip/active", headers=admin, json={"is_active": True})).json()["is_active"]

    # bad input
    bad = {**new, "prices": new["prices"][:2]}
    assert (await client.put("/admin/plan-tiers/vip", headers=admin, json=bad)).status_code == 422

    # a plan nobody bought can be deleted
    assert (await client.delete("/admin/plan-tiers/vip", headers=admin)).status_code == 200
    assert [t["tier"] for t in (await client.get("/admin/plan-tiers", headers=admin)).json()] == ["sello"]

    # trial limits are editable
    r = await client.put("/admin/free-limits", headers=admin, json={"limits": {**plans.FREE_LIMITS, "max_products": 40}})
    assert r.json()["limits"]["max_products"] == 40
    assert (await client.get("/admin/free-limits", headers=admin)).json()["limits"]["max_products"] == 40


async def test_change_plan_block_and_expiry(client):
    admin = await login(client, "09120000000")
    seller = await login(client, "09128880001")
    uid = await _user_id("09128880001")
    monthly, yearly = sorted((p for p in (await client.get("/plans")).json() if p["tier"] == "sello"),
                             key=lambda p: p["duration_months"])[::2]
    me = (await client.get("/me/subscription", headers=seller)).json()
    assert me["status"] == "free" and me["limits"]["monthly_chat_messages"] == plans.FREE_LIMITS["monthly_chat_messages"]

    assert (await client.post(f"/admin/users/{uid}/subscriptions", headers=admin, json={"plan_id": monthly["id"]})).status_code == 200
    # a renewal stacks after the current month
    await client.post(f"/admin/users/{uid}/subscriptions", headers=admin, json={"plan_id": monthly["id"]})
    subs = (await client.get(f"/admin/users/{uid}", headers=admin)).json()["subscriptions"]
    assert [s["status"] for s in subs] == ["active", "active"] and subs[0]["starts_at"] > subs[1]["starts_at"]
    # changing the plan now ends both and starts the new one today
    r = await client.post(f"/admin/users/{uid}/subscriptions", headers=admin, json={"plan_id": yearly["id"], "mode": "now"})
    assert r.status_code == 200
    detail = (await client.get(f"/admin/users/{uid}", headers=admin)).json()
    assert [s["status"] for s in detail["subscriptions"]] == ["active", "cancelled", "cancelled"]
    assert detail["subscriptions"][0]["duration_months"] == 12 and detail["plan_status"] == "active"
    assert detail["limits"]["monthly_chat_messages"] == 2000 and "chat_messages" in detail["usage"]
    row = next(u for u in (await client.get("/admin/users", headers=admin)).json() if u["id"] == uid)
    assert row["plan_status"] == "active" and row["plan_tier"] == "sello"

    # blocked: can't sign in, no AI for their stores
    assert (await client.post(f"/admin/users/{uid}/active", headers=admin, json={"is_active": False})).status_code == 200
    assert (await client.get("/me/subscription", headers=seller)).status_code == 401
    async with SessionLocal() as db:
        u = await db.get(User, uid)
        assert await plans.plan_status(db, u) == "blocked"
        assert (await plans.limits_for(db, u))["monthly_chat_messages"] == 0
    assert (await client.post(f"/admin/users/{uid}/active", headers=admin, json={"is_active": True})).status_code == 200

    # the plan ends: trial limits for counts, but no AI at all
    async with SessionLocal() as db:
        for s in (await db.execute(Subscription.__table__.select().where(Subscription.user_id == uid))).all():
            sub = await db.get(Subscription, s.id)
            sub.ends_at = utcnow() - timedelta(days=1)
        await db.commit()
    me = (await client.get("/me/subscription", headers=seller)).json()
    assert me["status"] == "expired" and me["subscription"] is None
    assert me["limits"]["monthly_chat_messages"] == 0 and me["limits"]["monthly_ai_credits"] == 0
    r = await client.post("/stores", headers=seller, json={"name": "تمام‌شده"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    async with SessionLocal() as db:
        assert "تمام شده" in await plans.credit_problem(db, sid, "product_generate")
        assert await plans.credit_problem(db, sid, "chat") == ""  # the chat has its own gate


async def test_ai_credits_cap(client):
    seller = await login(client, "09128880002")
    sid = (await client.post("/stores", headers=seller, json={"name": "اعتبار"})).json()["id"]
    async with SessionLocal() as db:
        await plans.save_free_limits(db, {"monthly_ai_credits": 22})
        db.add_all([AiUsageLog(created_at=utcnow(), task="chat", store_id=sid, ok=True) for _ in range(50)])  # not credits
        db.add(AiUsageLog(created_at=utcnow(), task="product_generate", store_id=sid, ok=True))
        db.add(AiUsageLog(created_at=utcnow(), task="vision", store_id=sid, ok=False))  # failed calls are free
        await db.commit()
        uid = await _user_id("09128880002")
        assert await plans.credits_used(db, uid) == 1
        assert await plans.credit_problem(db, sid, "product_generate") == ""
        assert await plans.credit_problem(db, sid, "image_edit") == ""  # 1 + 20 ≤ 22
        db.add(AiUsageLog(created_at=utcnow(), task="image_edit", store_id=sid, ok=True))
        await db.commit()
        assert await plans.credits_used(db, uid) == 21
        assert "اعتبار AI" in await plans.credit_problem(db, sid, "image_edit")
        await plans.save_free_limits(db, {})
        await db.commit()


async def test_chat_stops_when_plan_has_ended(client, monkeypatch):
    h, sid, llm, *_ = await _setup(client, monkeypatch, "09125550077")
    monkeypatch.setattr(chat_engine, "chat_limit", lambda db, store: _zero())
    conv = await _real_conv(sid)
    async with SessionLocal() as db:
        c = await db.get(chat_engine.ChatConversation, conv)
        rt.add_message(db, c, role="customer", text="مانتو لینن موجوده؟")
        await db.commit()
    await client.put(f"/stores/{sid}/chat/agent", headers=h, json={"enabled": True})
    before = len(llm.calls)
    saved = await rt.answer(conv, send=False)
    assert saved[-1].source == "limit" and len(llm.calls) == before


async def _zero():
    return 0
