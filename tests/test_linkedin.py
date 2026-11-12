"""LinkedIn accounts (email + password or the li_at cookie): sign-in, posts with photos, pacing, pausing when
LinkedIn pushes back, and messages and comments answered by the chat assistant. LinkedIn is faked at the HTTP level."""

import io
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.platforms import linkedin as li
from app.platforms.base import PlatformError

APP = {"enabled": True, "daily_limit": 2, "gap_min_minutes": 90, "gap_max_minutes": 180, "query_ids": {}}
ME = "ACoAAA1111"


class Ctx:
    def __init__(self, payload=None, db=None):
        self.steps = []
        self.op = SimpleNamespace(payload=payload or {}, result={}, current_step="")
        self.db = db or SimpleNamespace()

    @asynccontextmanager
    async def step(self, key, title, request=None):
        self.steps.append(key)
        yield SimpleNamespace(response=None)


class FakeLinkedIn:
    """Answers the Voyager calls the adapter makes and records them."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.login_result = "PASS"
        self.conversations: list[dict] = []
        self.comments: list[dict] = []
        self.fail: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = {}
        if request.content and request.headers.get("content-type", "").startswith(("application/json", "text/plain")):
            body = json.loads(request.content)
        self.calls.append((request.method, path, body))
        for key, status in self.fail.items():
            if key in path:
                return httpx.Response(status, text="slow down")
        if path == "/uas/authenticate" and request.method == "GET":
            return httpx.Response(200, headers={"set-cookie": 'JSESSIONID="ajax:123"; Path=/'})
        if path == "/uas/authenticate":
            cookies = {"set-cookie": "li_at=AQEDAT_session_cookie_value_123; Path=/"} if self.login_result == "PASS" else {}
            return httpx.Response(200, json={"login_result": self.login_result}, headers=cookies)
        if path == "/feed/":
            return httpx.Response(200, text="<html/>", headers={"set-cookie": 'JSESSIONID="ajax:456"; Path=/'})
        if path == "/voyager/api/me":
            assert request.headers["csrf-token"].startswith("ajax:")
            return httpx.Response(200, json={"data": {"plainId": 42, "*miniProfile": f"urn:li:fs_miniProfile:{ME}"},
                                             "included": [{"firstName": "Ali", "lastName": "Shop", "publicIdentifier": "alishop",
                                                           "entityUrn": f"urn:li:fs_miniProfile:{ME}"}]})
        if path == "/voyager/api/voyagerVideoDashMediaUploadMetadata":
            n = sum(1 for c in self.calls if c[1] == path)
            return httpx.Response(200, json={"data": {"value": {"singleUploadUrl": f"https://www.linkedin.com/dms-uploads/{n}",
                                                                 "urn": f"urn:li:digitalmediaAsset:IMG{n}"}}})
        if path.startswith("/dms-uploads/"):
            assert request.headers["media-type-family"] == "STILLIMAGE"
            return httpx.Response(201)
        if path == "/voyager/api/contentcreation/normShares":
            return httpx.Response(201, json={"data": {"status": {"urn": "urn:li:share:7001",
                                                                 "toastCtaUrl": "/feed/update/urn:li:activity:7000/"}}})
        if path == "/voyager/api/voyagerMessagingGraphQL/graphql":
            return httpx.Response(200, json={"data": {"messengerConversationsBySyncToken": {"elements": self.conversations}}})
        if path == "/voyager/api/voyagerMessagingDashMessengerMessages":
            n = sum(1 for c in self.calls if c[1] == path)
            return httpx.Response(200, json={"value": {"entityUrn": f"urn:li:msg_message:(x,sent{n})",
                                                       "conversationUrn": "urn:li:msg_conversation:(x,2-new)"}})
        if path == "/voyager/api/feed/comments":
            return httpx.Response(200, json={"elements": self.comments})
        if path == "/voyager/api/voyagerSocialDashNormComments":
            return httpx.Response(201, json={})
        return httpx.Response(404, text="unknown " + path)


@pytest.fixture
def fake(monkeypatch):
    server = FakeLinkedIn()
    transport = httpx.MockTransport(server)

    def client(self, timeout=60):
        return httpx.AsyncClient(transport=transport, follow_redirects=False)

    async def route(ctx, channel):
        return "socks5h://127.0.0.1:1080"

    async def app(db):
        return APP

    monkeypatch.setattr(li.Session, "_client", client)
    monkeypatch.setattr(li, "_route", route)
    monkeypatch.setattr(li, "linkedin_app", app)
    monkeypatch.setattr("app.services.integrations.linkedin_app", app)
    monkeypatch.setattr("app.services.li_chat.linkedin_app", app)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    return server


async def _no_sleep(*_a, **_k):
    return None


@pytest.fixture
def media(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "media_dir", str(tmp_path))
    for n in ("a.png", "b.png"):
        buf = io.BytesIO()
        Image.new("RGB", (30, 30), "red").save(buf, "PNG")
        (tmp_path / n).write_bytes(buf.getvalue())
    return tmp_path


def _channel(**cfg):
    creds = {"email": "a@b.c", "password": "pw", "user_agent": li.USER_AGENTS[0],
             "cookies": {"li_at": "AQEDAT_session_cookie_value_123", "JSESSIONID": '"ajax:123"'}}
    return SimpleNamespace(id=9, platform="linkedin", status="connected", credentials_encrypted=encrypt_credentials(creds),
                           config={"mode": "session", "profile_id": ME, **cfg})


async def test_connect_with_email_and_password(fake):
    channel = SimpleNamespace(id=1, credentials_encrypted=None, config={})
    res = await li.linkedin.connect(Ctx(), channel, {"email": "Shop@Mail.com", "password": "pw"})
    assert res["display_name"] == "Ali Shop" and res["external_account_id"] == ME
    assert res["config"]["public_id"] == "alishop" and res["config"]["mode"] == "session"
    assert res["credentials"]["cookies"]["li_at"].startswith("AQEDAT")
    assert res["credentials"]["user_agent"] in li.USER_AGENTS  # the same browser from now on
    login = next(c for c in fake.calls if c[0] == "POST" and c[1] == "/uas/authenticate")
    assert login[2] == {}  # form data, not JSON


async def test_a_security_check_asks_for_the_cookie(fake):
    fake.login_result = "CHALLENGE"
    channel = SimpleNamespace(id=1, credentials_encrypted=None, config={})
    with pytest.raises(PlatformError) as e:
        await li.linkedin.connect(Ctx(), channel, {"email": "a@b.c", "password": "pw"})
    assert e.value.code == "li_challenge" and e.value.suggested_action["type"] == "cookie"
    # The cookie path works without a password.
    res = await li.linkedin.connect(Ctx(), channel, {"li_at": "li_at=AQEDAT_from_the_browser_cookie; Path=/"})
    assert res["credentials"]["cookies"]["li_at"] == "AQEDAT_from_the_browser_cookie"
    assert res["credentials"]["cookies"]["JSESSIONID"] == '"ajax:456"' and res["credentials"]["by_cookie"]


async def test_post_with_photos(fake, media):
    channel = _channel()
    product = SimpleNamespace(title="کیف چرم", description="دست‌دوز", price=250000, post_caption="",
                              images=[SimpleNamespace(url="/media/a.png"), SimpleNamespace(url="/media/b.png")])
    store = SimpleNamespace(settings={"show_price_in_caption": True})
    ctx = Ctx()
    res = await li.linkedin.publish(ctx, channel, product, store)
    assert res.remote_id == "urn:li:activity:7000" and res.remote_url.endswith("/feed/update/urn:li:activity:7000/")
    share = next(c[2] for c in fake.calls if c[1].endswith("/normShares"))
    assert [m["mediaUrn"] for m in share["media"]] == ["urn:li:digitalmediaAsset:IMG1", "urn:li:digitalmediaAsset:IMG2"]
    assert "کیف چرم" in share["commentaryV2"]["text"]
    assert ctx.steps == ["images", "publish"]


async def test_too_many_requests_pauses_the_account(fake, media):
    channel = _channel()
    fake.fail = {"normShares": 429}
    product = SimpleNamespace(title="کیف", description="", price=0, post_caption="", images=[])
    with pytest.raises(PlatformError) as e:
        await li.linkedin.publish(Ctx(), channel, product, SimpleNamespace(settings={}))
    assert e.value.code == "li_wait" and channel.config["paused_until"]


async def test_expired_session_signs_in_once_again(fake, media):
    channel = _channel()
    calls = {"n": 0}
    real = li.Session.me

    async def me(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PlatformError("expired", "expired")
        return await real(self)

    li.Session.me = me
    try:
        await li.linkedin.health(Ctx(), channel)
    finally:
        li.Session.me = real
    assert calls["n"] == 2
    creds = decrypt_credentials(channel.credentials_encrypted)
    assert creds["relogin_at"] and creds["cookies"]["li_at"].startswith("AQEDAT")


async def test_paced_publish_waits_for_its_slot(client, fake):
    from app.core.db import SessionLocal
    from app.models import Channel, Operation
    from tests.conftest import drain, login

    h = await login(client, "09128880461")
    sid = (await client.post("/stores", json={"name": "لینکدین"}, headers=h)).json()["id"]
    p = (await client.post(f"/stores/{sid}/products", headers=h, json={"title": "کیف", "price": 1000})).json()
    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="linkedin", status="connected", display_name="Ali",
                     credentials_encrypted=_channel().credentials_encrypted,
                     config={"mode": "session", "profile_id": ME, "next_at": (utcnow() + timedelta(minutes=60)).isoformat()})
        db.add(ch)
        await db.commit()
        cid = ch.id
    assert (await client.post(f"/products/{p['id']}/publish", headers=h,
                              json={"channel_ids": [cid], "format": "story"})).status_code == 422
    r = await client.post(f"/products/{p['id']}/publish", headers=h, json={"channel_ids": [cid]})
    op_id = r.json()[0]["id"]
    await drain()
    async with SessionLocal() as db:
        op = await db.get(Operation, op_id)
        assert op.status == "queued" and op.run_after is not None and "در صف" in op.current_step
        ch = await db.get(Channel, cid)
        ch.config = {**ch.config, "next_at": None}
        op.run_after = None
        await db.commit()
    await drain()
    async with SessionLocal() as db:
        op = await db.get(Operation, op_id)
        assert op.status == "succeeded", op.error_fa if hasattr(op, "error_fa") else op
        assert op.result["remote_id"] == "urn:li:activity:7000"
        assert (await db.get(Channel, cid)).config["count"] == 1


def _conv(urn, other_id, msgs, name="Sara Ahmadi"):
    return {"entityUrn": urn, "lastActivityAt": max(m["deliveredAt"] for m in msgs), "groupChat": False,
            "conversationParticipants": [
                {"hostIdentityUrn": f"urn:li:fsd_profile:{ME}"},
                {"hostIdentityUrn": f"urn:li:fsd_profile:{other_id}",
                 "participantType": {"member": {"firstName": {"text": name.split()[0]}, "lastName": {"text": name.split()[1]},
                                                "profileUrl": f"https://www.linkedin.com/in/{other_id.lower()}"}}}],
            "messages": {"elements": msgs}}


def _msg(mid, text, sender, minutes=1):
    return {"entityUrn": f"urn:li:msg_message:(x,{mid})", "body": {"text": text},
            "sender": {"hostIdentityUrn": f"urn:li:fsd_profile:{sender}"},
            "deliveredAt": int((utcnow() + timedelta(minutes=minutes)).timestamp() * 1000)}


async def test_messages_and_comments_reach_the_assistant(client, fake):
    from sqlalchemy import select

    from app.core.db import SessionLocal
    from app.models import Channel, ChannelListing, ChatBot, ChatConversation, ChatMessage, Product
    from app.services import li_chat
    from tests.conftest import login

    h = await login(client, "09128880462")
    sid = (await client.post("/stores", json={"name": "پیام لینکدین"}, headers=h)).json()["id"]
    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="linkedin", status="connected", display_name="Ali Shop",
                     credentials_encrypted=_channel().credentials_encrypted,
                     config={"mode": "session", "profile_id": ME, "public_id": "alishop"})
        p = Product(store_id=sid, title="کیف چرم", price=1000)
        db.add_all([ch, p])
        await db.flush()
        db.add(ChannelListing(product_id=p.id, channel_id=ch.id, status="published", remote_id="urn:li:activity:7000",
                              published_at=utcnow()))
        await db.commit()
        cid = ch.id
    r = await client.put(f"/channels/{cid}/linkedin/inbox", headers=h, json={"dm": True, "comments": True})
    assert r.status_code == 200 and r.json()["config"]["comment_reply"] is True
    assert (await client.get(f"/stores/{sid}/chat/bots", headers=h)).json() == []  # set on the channels page
    async with SessionLocal() as db:
        bot_id = (await li_chat.bot_for(db, await db.get(Channel, cid))).id

    fake.conversations = [_conv("urn:li:msg_conversation:(x,2-abc)", "ACoAAB2222",
                                [_msg("old", "قدیمی", "ACoAAB2222", -60), _msg("m1", "سلام، قیمت کیف؟", "ACoAAB2222")])]
    fake.comments = [{"urn": "urn:li:comment:(activity:7000,555)", "entityUrn": "urn:li:fs_objectComment:(555,activity:7000)",
                      "commentV2": {"text": "موجوده؟"}, "createdTime": int((utcnow() + timedelta(minutes=1)).timestamp() * 1000),
                      "commenter": {"com.linkedin.voyager.feed.MemberActor": {"miniProfile": {
                          "entityUrn": "urn:li:fs_miniProfile:ACoAAC3333", "firstName": "Reza", "lastName": "K",
                          "publicIdentifier": "rezak"}}}}]
    conv_ids = await li_chat.poll_once(bot_id, comments_due=True)
    assert len(conv_ids) == 2
    assert await li_chat.poll_once(bot_id, comments_due=True) == []  # the same messages again: nothing new
    async with SessionLocal() as db:
        texts = (await db.execute(select(ChatMessage.text).where(ChatMessage.conversation_id.in_(conv_ids))
                                  .order_by(ChatMessage.id))).scalars().all()
        assert "سلام، قیمت کیف؟" in texts and "قدیمی" not in texts
        assert any("کامنت زیر پست لینکدین محصول «کیف چرم»" in t for t in texts)
        convs = {c.external_chat_id: c for c in (await db.execute(
            select(ChatConversation).where(ChatConversation.id.in_(conv_ids)))).scalars()}
        assert convs["ACoAAB2222"].customer_name == "Sara Ahmadi"

        # The assistant answers: the message in its conversation, the comment under the comment.
        bot = await db.get(ChatBot, bot_id)
        rows = [SimpleNamespace(external_id="")]
        await li_chat.deliver(db, bot, convs["ACoAAB2222"], [SimpleNamespace(text="۱۲۰ هزار تومان", image_url="",
                                                                              buttons=[["ثبت سفارش"]])], rows)
        dm = [c[2] for c in fake.calls if c[1].endswith("/voyagerMessagingDashMessengerMessages")][-1]
        assert dm["message"]["conversationUrn"] == "urn:li:msg_conversation:(x,2-abc)"
        assert "• ثبت سفارش" in dm["message"]["body"]["text"] and rows[0].external_id.startswith("li:")
        await li_chat.deliver(db, bot, convs["ACoAAC3333"], [SimpleNamespace(text="بله موجوده", image_url="", buttons=None)],
                              [SimpleNamespace(external_id="")])
        reply = [c[2] for c in fake.calls if c[1].endswith("/voyagerSocialDashNormComments")][-1]
        assert reply["threadUrn"] == "urn:li:comment:(urn:li:activity:7000,555)"
        assert reply["commentary"]["text"] == "Reza عزیز، بله موجوده"
        await db.commit()

    # Our own sent message comes back in the inbox: not the seller typing. A message the seller typed: AI steps back.
    fake.conversations = [_conv("urn:li:msg_conversation:(x,2-abc)", "ACoAAB2222", [
        _msg("m1", "سلام، قیمت کیف؟", "ACoAAB2222"), _msg("sent1", "۱۲۰ هزار تومان\n\n• ثبت سفارش", ME, 2)])]
    assert await li_chat.poll_once(bot_id, comments_due=False) == []
    async with SessionLocal() as db:
        assert (await db.get(ChatConversation, convs["ACoAAB2222"].id)).status != "human"
    fake.conversations[0]["messages"]["elements"].append(_msg("me2", "خودم جواب می‌دم", ME, 3))
    fake.conversations[0]["lastActivityAt"] = fake.conversations[0]["messages"]["elements"][-1]["deliveredAt"]
    assert await li_chat.poll_once(bot_id, comments_due=False) == []
    async with SessionLocal() as db:
        assert (await db.get(ChatConversation, convs["ACoAAB2222"].id)).status == "human"


async def test_linkedin_on_the_chat_connections_switch(client, fake):
    from app.core.db import SessionLocal
    from app.models import Channel

    from tests.conftest import login

    h = await login(client, "09128880463")
    sid = (await client.post("/stores", json={"name": "سوییچ لینکدین"}, headers=h)).json()["id"]
    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="linkedin", status="connected", display_name="Ali Shop",
                     credentials_encrypted=_channel().credentials_encrypted, config={"mode": "session", "profile_id": ME})
        db.add(ch)
        await db.commit()
        cid = ch.id
    rows = (await client.get(f"/stores/{sid}/chat/channels", headers=h)).json()
    row = next(r for r in rows if r["channel_id"] == cid)
    assert row["platform"] == "linkedin" and row["available"] and not row["active"]
    assert [f["key"] for f in row["features"]] == ["dm", "comments"]
    r = await client.put(f"/chat/channels/{cid}", headers=h, json={"dm": True, "comments": False, "comment_text": ""})
    assert r.status_code == 200, r.text
    assert r.json()["active"] and r.json()["dm"] and not r.json()["comments"]
    assert (await client.get(f"/stores/{sid}/chat/bots", headers=h)).json() == []
