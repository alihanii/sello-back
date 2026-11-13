from datetime import timedelta

import httpx
import jwt
import pytest

from app.core.config import settings
from app.core.db import SessionLocal, utcnow
from app.main import app
from app.models import AuthSession


@pytest.fixture
async def fresh(client):  # `client` makes sure the database is set up
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1") as c:
        yield c


async def otp_login(c, phone):
    r = await c.post("/auth/otp/send", json={"phone": phone})
    assert r.status_code == 200, r.text
    r = await c.post("/auth/otp/verify", json={"phone": phone, "code": r.json()["dev_code"]})
    assert r.status_code == 200, r.text
    return r


def rt(c):
    return c.cookies.get(settings.refresh_cookie_name)


async def test_otp_login_sets_httponly_refresh_cookie_and_short_access_token(fresh):
    r = await otp_login(fresh, "09130000001")
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api/v1/auth" in cookie
    body = r.json()
    assert body["expires_in"] == settings.jwt_access_minutes * 60 and body["has_password"] is False
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert claims["typ"] == "access" and claims["sid"]
    assert (await fresh.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})).status_code == 200


async def test_refresh_rotates_and_reuse_revokes_the_session(fresh):
    await otp_login(fresh, "09130000002")
    first = rt(fresh)
    r = await fresh.post("/auth/refresh")
    assert r.status_code == 200, r.text
    second = rt(fresh)
    assert second and second != first
    access = r.json()["access_token"]
    # Within the grace window the replaced token still yields an access token (two tabs at once)...
    r = await fresh.post("/auth/refresh", cookies={settings.refresh_cookie_name: first})
    assert r.status_code == 200
    # ...after it, the replaced token coming back means theft: the whole session is closed.
    async with SessionLocal() as db:
        s = await db.get(AuthSession, jwt.decode(access, options={"verify_signature": False})["sid"])
        s.rotated_at = utcnow() - timedelta(minutes=5)
        await db.commit()
    fresh.cookies.clear()
    r = await fresh.post("/auth/refresh", cookies={settings.refresh_cookie_name: first})
    assert r.status_code == 401
    r = await fresh.post("/auth/refresh", cookies={settings.refresh_cookie_name: second})
    assert r.status_code == 401
    assert (await fresh.get("/auth/me", headers={"Authorization": f"Bearer {access}"})).status_code == 401


async def test_logout_revokes_access_too(fresh):
    r = await otp_login(fresh, "09130000003")
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert (await fresh.post("/auth/logout")).status_code == 200
    assert (await fresh.get("/auth/me", headers=h)).status_code == 401
    assert (await fresh.post("/auth/refresh")).status_code == 401


async def test_refresh_refuses_other_sites(fresh):
    await otp_login(fresh, "09130000004")
    r = await fresh.post("/auth/refresh", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


async def test_expired_session_needs_sign_in(fresh):
    r = await otp_login(fresh, "09130000005")
    sid = jwt.decode(r.json()["access_token"], options={"verify_signature": False})["sid"]
    async with SessionLocal() as db:
        s = await db.get(AuthSession, sid)
        s.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    assert (await fresh.post("/auth/refresh")).status_code == 401


async def test_password_flow_lookup_login_change_and_lock(fresh):
    phone = "09130000006"
    assert (await fresh.post("/auth/lookup", json={"phone": phone})).json() == {
        "exists": False, "has_password": False, "method": "otp"}
    r = await otp_login(fresh, phone)
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert (await fresh.post("/auth/lookup", json={"phone": phone})).json()["method"] == "otp"
    assert (await fresh.post("/auth/password", headers=h, json={"password": "short"})).status_code == 422
    assert (await fresh.post("/auth/password", headers=h, json={"password": "kaktus-1405"})).status_code == 200
    assert (await fresh.get("/auth/me", headers=h)).json()["has_password"] is True
    assert (await fresh.post("/auth/lookup", json={"phone": "۰۹۱۳۰۰۰۰۰۰۶"})).json() == {
        "exists": True, "has_password": True, "method": "password"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1") as other:
        r = await other.post("/auth/password/login", json={"phone": phone, "password": "kaktus-1405"})
        assert r.status_code == 200, r.text
        assert rt(other)
        h2 = {"Authorization": f"Bearer {r.json()['access_token']}"}
        # Changing it from the password device needs the current password, and signs other devices out.
        assert (await other.post("/auth/password", headers=h2, json={"password": "new-pass-99"})).status_code == 400
        r = await other.post("/auth/password", headers=h2, json={"password": "new-pass-99", "current_password": "kaktus-1405"})
        assert r.status_code == 200
        assert (await other.get("/auth/me", headers=h2)).status_code == 200
    assert (await fresh.get("/auth/me", headers=h)).status_code == 401

    for _ in range(5):
        r = await fresh.post("/auth/password/login", json={"phone": phone, "password": "wrong-one-1"})
        assert r.status_code == 400
    r = await fresh.post("/auth/password/login", json={"phone": phone, "password": "new-pass-99"})
    assert r.status_code == 429  # locked for a while...
    await otp_login(fresh, phone)  # ...but a code still works and clears the lock
    assert (await fresh.post("/auth/password/login", json={"phone": phone, "password": "new-pass-99"})).status_code == 200


async def test_unknown_phone_password_login_is_a_plain_error(fresh):
    r = await fresh.post("/auth/password/login", json={"phone": "09130000099", "password": "whatever-1"})
    assert r.status_code == 400 and "اشتباه" in r.json()["detail"]["message"]


async def test_old_token_upgrades_to_a_session_once(fresh):
    await otp_login(fresh, "09130000007")
    me = (await fresh.post("/auth/refresh")).json()
    uid = jwt.decode(me["access_token"], options={"verify_signature": False})["sub"]
    legacy = jwt.encode({"sub": uid, "role": "seller", "exp": utcnow() + timedelta(hours=1)}, settings.jwt_secret)
    fresh.cookies.clear()
    # The old token isn't an access token any more...
    assert (await fresh.get("/auth/me", headers={"Authorization": f"Bearer {legacy}"})).status_code == 401
    # ...but trades for a session.
    r = await fresh.post("/auth/session/upgrade", headers={"Authorization": f"Bearer {legacy}"})
    assert r.status_code == 200 and rt(fresh)
    assert (await fresh.post("/auth/refresh")).status_code == 200


async def test_codes_go_by_kavenegar_when_configured(fresh, monkeypatch):
    from app.routers import auth

    sent = []

    async def fake_send(cfg, phone, code):
        sent.append((cfg["template"], phone, code))

    async def cfg(db):
        return {"enabled": True, "provider": "kavenegar", "api_key": "k" * 20, "template": "foroshify-login",
                "sender": "", "ready": True}

    monkeypatch.setattr(auth, "send_code", fake_send)
    monkeypatch.setattr(auth, "sms_config", cfg)
    r = await fresh.post("/auth/otp/send", json={"phone": "09130000008"})
    assert r.status_code == 200 and "dev_code" not in r.json()
    assert sent and sent[0][:2] == ("foroshify-login", "09130000008")
    r = await fresh.post("/auth/otp/verify", json={"phone": "09130000008", "code": sent[0][2]})
    assert r.status_code == 200


async def test_no_sms_and_no_dev_mode_refuses(fresh, monkeypatch):
    monkeypatch.setattr(settings, "otp_dev_mode", False)
    r = await fresh.post("/auth/otp/send", json={"phone": "09130000009"})
    assert r.status_code == 503


async def test_kavenegar_request_shape(monkeypatch):
    from app.services import sms

    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["body"] = str(request.url), request.content.decode()
        return httpx.Response(200, json={"return": {"status": 200, "message": "تایید شد"}, "entries": [{}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(sms.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    await sms.send_code({"api_key": "KEY123", "template": "foroshify-login"}, "09121234567", "456789")
    assert seen["url"] == "https://api.kavenegar.com/v1/KEY123/verify/lookup.json"
    assert "receptor=09121234567" in seen["body"] and "token=456789" in seen["body"] and "template=foroshify-login" in seen["body"]

    def bad(request):
        return httpx.Response(418, json={"return": {"status": 418, "message": "credit"}, "entries": None})

    monkeypatch.setattr(sms.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(bad)))
    with pytest.raises(sms.SmsError, match="اعتبار"):
        await sms.send_code({"api_key": "KEY123", "template": "t"}, "09121234567", "1")


async def test_ip_limit_on_code_sends(fresh, monkeypatch):
    from app.core import ratelimit

    monkeypatch.setattr(settings, "ratelimit_enabled", True)
    ratelimit.reset()
    codes = [(await fresh.post("/auth/otp/send", json={"phone": f"0913100{i:04d}"})).status_code for i in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    ratelimit.reset()


async def test_wrong_codes_wait_40_seconds_then_lock_for_an_hour(fresh):
    from app.models import OtpThrottle

    phone = "09130000010"
    sent = await fresh.post("/auth/otp/send", json={"phone": phone})
    code = sent.json()["dev_code"]
    wrong = "000000" if code != "000000" else "111111"

    async def clear_wait():
        async with SessionLocal() as db:
            t = await db.get(OtpThrottle, phone)
            t.next_try_at = None
            await db.commit()

    r = await fresh.post("/auth/otp/verify", json={"phone": phone, "code": wrong})
    assert r.status_code == 400 and r.json()["detail"]["retry_after"] == 40
    # Right away, even the right code has to wait.
    r = await fresh.post("/auth/otp/verify", json={"phone": phone, "code": code})
    assert r.status_code == 429 and 1 <= r.json()["detail"]["retry_after"] <= 40
    for _ in range(3):
        await clear_wait()
        assert (await fresh.post("/auth/otp/verify", json={"phone": phone, "code": wrong})).status_code == 400
    await clear_wait()
    r = await fresh.post("/auth/otp/verify", json={"phone": phone, "code": wrong})  # the fifth in a row
    assert r.status_code == 429 and r.json()["detail"]["retry_after"] == 3600
    # Locked: neither the right code nor a new code helps.
    assert (await fresh.post("/auth/otp/verify", json={"phone": phone, "code": code})).status_code == 429
    assert (await fresh.post("/auth/otp/send", json={"phone": phone})).status_code == 429
    async with SessionLocal() as db:
        t = await db.get(OtpThrottle, phone)
        t.locked_until = utcnow() - timedelta(seconds=1)
        await db.commit()
    r = await fresh.post("/auth/otp/send", json={"phone": phone})
    assert r.status_code == 200
    r = await fresh.post("/auth/otp/verify", json={"phone": phone, "code": r.json()["dev_code"]})
    assert r.status_code == 200


async def test_a_right_code_clears_the_misses(fresh):
    from app.models import OtpThrottle

    phone = "09130000011"
    code = (await fresh.post("/auth/otp/send", json={"phone": phone})).json()["dev_code"]
    await fresh.post("/auth/otp/verify", json={"phone": phone, "code": "000000" if code != "000000" else "111111"})
    async with SessionLocal() as db:
        t = await db.get(OtpThrottle, phone)
        t.next_try_at = None
        await db.commit()
    assert (await fresh.post("/auth/otp/verify", json={"phone": phone, "code": code})).status_code == 200
    async with SessionLocal() as db:
        t = await db.get(OtpThrottle, phone)
        assert t.fails == 0 and t.next_try_at is None
