"""Instagram accounts connected with username + password (instagrapi): login with a 2FA code, posts, albums,
stories, reels from the product video, pacing, and pausing the account when Instagram pushes back."""

import asyncio
import io
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.db import utcnow
from app.core.security import decrypt_credentials, encrypt_credentials
from app.platforms import instagram as ig
from app.platforms import instagram_private as igp
from app.platforms.base import PlatformError

APP = {"enabled": True, "app_id": "", "app_secret": "", "proxy": "", "oauth_ready": False, "method": "login",
       "daily_limit": 3, "gap_min_minutes": 20, "gap_max_minutes": 40, "redirect_uri": ""}


class Ctx:
    def __init__(self, payload=None):
        self.steps = []
        self.op = SimpleNamespace(payload=payload or {}, result={}, current_step="")
        self.db = SimpleNamespace(commit=self._commit)

    async def _commit(self):
        pass

    @asynccontextmanager
    async def step(self, key, title, request=None):
        self.steps.append(key)
        yield SimpleNamespace(response=None)


class FakeClient:
    """Records what the adapter asks instagrapi to do."""

    def __init__(self, two_factor=False):
        self.calls = []
        self.user_id = 555
        self.two_factor = two_factor
        self.challenge_code_handler = None

    def login(self, username, password, verification_code=""):
        from instagrapi.exceptions import TwoFactorRequired

        self.calls.append(("login", username, verification_code))
        if self.two_factor and not verification_code:
            raise TwoFactorRequired("2fa")

    def account_info(self):
        return SimpleNamespace(username="myshop", is_business=False)

    def get_settings(self):
        return {"uuids": {"phone_id": "p1"}, "cookies": {"sessionid": "s"}}

    def _media(self, kind, *args, **kw):
        self.calls.append((kind, args, kw))
        return SimpleNamespace(pk=1000 + len(self.calls), code="Cabc")

    def photo_upload(self, *a, **k):
        return self._media("photo", *a, **k)

    def album_upload(self, *a, **k):
        return self._media("album", *a, **k)

    def photo_upload_to_story(self, *a, **k):
        return self._media("photo_story", *a, **k)

    def video_upload_to_story(self, *a, **k):
        return self._media("video_story", *a, **k)

    def clip_upload(self, *a, **k):
        return self._media("clip", *a, **k)


@pytest.fixture
def media(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "media_dir", str(tmp_path))
    for n in ("a.png", "b.webp"):
        buf = io.BytesIO()
        Image.new("RGB", (30, 30), "blue").save(buf, "PNG" if n.endswith("png") else "WEBP")
        (tmp_path / n).write_bytes(buf.getvalue())
    return tmp_path


def _product(**kw):
    base = dict(title="کیف چرم", description="دست‌دوز", price=250000, compare_at_price=0, post_caption="", stock_qty=1,
                attributes={}, tags=[], video_url="",
                images=[SimpleNamespace(url="/media/a.png"), SimpleNamespace(url="/media/b.webp")])
    return SimpleNamespace(**{**base, **kw})


def _channel(fake):
    creds = {"username": "myshop", "password": "pw", "settings": fake.get_settings()}
    return SimpleNamespace(id=7, platform="instagram", credentials_encrypted=encrypt_credentials(creds),
                           config={"mode": "private", "ig_user_id": "555", "username": "myshop", "proxy_node_id": 1})


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()

    async def route(ctx, channel):
        return "socks5h://127.0.0.1:1080"

    monkeypatch.setattr(igp, "_route", route)
    monkeypatch.setattr(igp, "_client", lambda proxy, saved: client)

    async def app(db):
        return APP

    monkeypatch.setattr(ig, "instagram_app", app)
    return client


async def test_connect_waits_for_the_two_factor_code(fake):
    fake.two_factor = True
    channel = SimpleNamespace(id=11, credentials_encrypted=None, config={})
    ctx = Ctx()

    async def type_code():
        while not igp.waiting_for_code(11) or not ctx.op.result.get("needs_code"):
            await asyncio.sleep(0.01)
        assert ctx.op.result["needs_code"] == "2fa"
        assert igp.submit_code(11, "۱۲۳ 456")  # Persian digits and spaces are fine

    typer = asyncio.create_task(type_code())
    res = await ig.instagram.connect(ctx, channel, {"username": "@MyShop", "password": "pw"})
    await typer
    assert fake.calls == [("login", "myshop", ""), ("login", "myshop", "123456")]
    assert res["display_name"] == "@myshop" and res["config"]["mode"] == "private"
    assert res["credentials"]["settings"]["uuids"]["phone_id"] == "p1"  # the same device next time
    assert not igp.waiting_for_code(11) and ctx.op.result == {}


async def test_post_album_story_and_validation(fake, media):
    channel = _channel(fake)
    out = await ig.instagram.publish(Ctx(), channel, _product(), SimpleNamespace(settings={}))
    kind, args, _ = fake.calls[-1]
    assert kind == "album" and len(args[0]) == 2 and args[1].startswith("کیف چرم")
    assert all(p.suffix == ".jpg" for p in args[0])
    assert out.remote_url == "https://www.instagram.com/p/Cabc/"

    out = await ig.instagram.publish(Ctx(), channel, _product(images=[SimpleNamespace(url="/media/a.png")]),
                                     SimpleNamespace(settings={}))
    assert fake.calls[-1][0] == "photo"

    out = await ig.instagram.publish(Ctx({"format": "story"}), channel, _product(), SimpleNamespace(settings={}))
    assert fake.calls[-1][0] == "photo_story" and out.remote_url.startswith("https://www.instagram.com/stories/myshop/")

    assert ig.instagram.validate_publish(channel, _product(), "reel")  # no video
    assert ig.instagram.validate_publish(channel, _product(images=[]), "post")
    assert not ig.instagram.validate_publish(channel, _product(), "story")
    official = SimpleNamespace(config={"ig_user_id": "1"})
    assert ig.instagram.validate_publish(official, _product(), "story")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
async def test_reel_from_the_product_video(fake, media):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x568:rate=25", "-t", "2",
                    "-pix_fmt", "yuv420p", str(media / "v.mov")], check=True)
    channel = _channel(fake)
    out = await ig.instagram.publish(Ctx({"format": "reel"}), channel, _product(video_url="/media/v.mov"),
                                     SimpleNamespace(settings={}))
    kind, args, kw = fake.calls[-1]
    assert kind == "clip" and args[0].suffix == ".mp4" and kw["thumbnail"].suffix == ".jpg"
    assert out.remote_url == "https://www.instagram.com/reel/Cabc/"
    await ig.instagram.publish(Ctx({"format": "story"}), channel, _product(video_url="/media/v.mov"),
                               SimpleNamespace(settings={}))
    assert fake.calls[-1][0] == "video_story"


async def test_feedback_required_pauses_the_account(fake, media):
    from instagrapi.exceptions import FeedbackRequired

    def blocked(*a, **k):
        raise FeedbackRequired("feedback_required")

    fake.album_upload = blocked
    channel = _channel(fake)
    with pytest.raises(PlatformError) as e:
        await ig.instagram.publish(Ctx(), channel, _product(), SimpleNamespace(settings={}))
    assert e.value.code == "ig_feedback"
    assert igp.next_slot(channel, APP, utcnow()) > utcnow() + timedelta(hours=23)


def test_pacing_gap_and_daily_cap():
    ch = SimpleNamespace(config={})
    now = utcnow()
    assert igp.next_slot(ch, APP, now) is None
    igp.reserve(ch, APP, now)
    nxt = igp.next_slot(ch, APP, now)
    assert now + timedelta(minutes=19) < nxt <= now + timedelta(minutes=41)
    later = now + timedelta(minutes=41)
    assert igp.next_slot(ch, APP, later) is None
    igp.reserve(ch, APP, later)
    igp.reserve(ch, APP, later + timedelta(minutes=41))
    assert ch.config["count"] == 3
    # The cap (3) is reached: nothing more today, even after the gap.
    assert igp.next_slot(ch, APP, later + timedelta(minutes=90)) > later + timedelta(minutes=90)


async def test_paced_publish_goes_back_to_the_queue(client, monkeypatch):
    """A publish inside the account's gap waits for its slot instead of running; a story for Telegram is refused."""
    from app.core.db import SessionLocal
    from app.models import Channel, Operation
    from tests.conftest import drain, login

    h = await login(client, "09128880451")
    sid = (await client.post("/stores", json={"name": "پیج نمونه"}, headers=h)).json()["id"]
    p = (await client.post(f"/stores/{sid}/products", headers=h, json={"title": "کیف", "price": 1000,
                                                                      "images": ["/media/a.png"]})).json()
    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="instagram", status="connected", display_name="@x",
                     credentials_encrypted=encrypt_credentials({"username": "x", "password": "p", "settings": {"a": 1}}),
                     config={"mode": "private", "ig_user_id": "1", "username": "x",
                             "next_at": (utcnow() + timedelta(minutes=30)).isoformat()})
        tg = Channel(store_id=sid, platform="telegram", status="connected", config={})
        db.add_all([ch, tg])
        await db.commit()
        cid, tid = ch.id, tg.id

    async def app(db):
        return APP

    monkeypatch.setattr("app.services.integrations.instagram_app", app)
    r = await client.post(f"/products/{p['id']}/publish", headers=h, json={"channel_ids": [cid], "format": "story"})
    assert r.status_code == 202, r.text
    op_id = r.json()[0]["id"]
    again = await client.post(f"/products/{p['id']}/publish", headers=h, json={"channel_ids": [cid], "format": "story"})
    assert again.json() == []  # the same story is already queued
    assert (await client.post(f"/products/{p['id']}/publish", headers=h,
                              json={"channel_ids": [tid], "format": "reel"})).status_code == 422
    await drain()
    async with SessionLocal() as db:
        op = await db.get(Operation, op_id)
        assert op.status == "queued" and op.run_after is not None and "در صف" in op.current_step


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
async def test_upload_a_product_video(client, tmp_path):
    from tests.conftest import login

    h = await login(client, "09128880452")
    sid = (await client.post("/stores", json={"name": "ویدیو"}, headers=h)).json()["id"]
    path = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25", "-t", "1",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    r = await client.post(f"/stores/{sid}/uploads/video", headers=h,
                          files={"file": ("v.mp4", path.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["url"].startswith(f"/media/{sid}/") and v["url"].endswith(".mp4") and v["width"] == 320
    bad = await client.post(f"/stores/{sid}/uploads/video", headers=h, files={"file": ("v.mp4", b"not a video", "video/mp4")})
    assert bad.status_code == 422
    p = (await client.post(f"/stores/{sid}/products", headers=h, json={"title": "ویدیودار", "video_url": v["url"]})).json()
    assert p["video_url"] == v["url"]
    r = await client.patch(f"/products/{p['id']}", headers=h, json={"video_url": "/media/999/x.mp4"})
    assert r.status_code == 422
    r = await client.patch(f"/products/{p['id']}", headers=h, json={"video_url": ""})
    assert r.json()["video_url"] == ""


# ---------- DMs and comments answered by the assistant ----------

def _thread(tid, msgs, user_pk="900", username="customer1", pending=False):
    users = [SimpleNamespace(pk=user_pk, username=username, full_name="مشتری")]
    return SimpleNamespace(id=tid, is_group=False, users=users, messages=msgs, pending=pending,
                           last_activity_at=max(m.timestamp for m in msgs))


def _dm(mid, text, user_id="900", minutes=1, **kw):
    base = dict(id=mid, user_id=user_id, thread_id=None, timestamp=utcnow() + timedelta(minutes=minutes), item_type="text",
                text=text, media=None, media_share=None, clip=None, reel_share=None)
    return SimpleNamespace(**{**base, **kw})


async def test_dms_and_comments_reach_the_assistant(client, fake, monkeypatch):
    from sqlalchemy import select

    from app.core.db import SessionLocal
    from app.models import Channel, ChannelListing, ChatConversation, ChatMessage, Product
    from app.services import ig_chat
    from tests.conftest import login

    h = await login(client, "09128880453")
    sid = (await client.post("/stores", json={"name": "دایرکت"}, headers=h)).json()["id"]
    async with SessionLocal() as db:
        ch = Channel(store_id=sid, platform="instagram", status="connected", display_name="@myshop",
                     credentials_encrypted=encrypt_credentials({"username": "myshop", "password": "pw", "settings": {"a": 1}}),
                     config={"mode": "private", "ig_user_id": "555", "username": "myshop"})
        p = Product(store_id=sid, title="کیف چرم", sku="K7", price=1000)
        db.add_all([ch, p])
        await db.flush()
        db.add(ChannelListing(product_id=p.id, channel_id=ch.id, status="published", remote_id="3100", published_at=utcnow()))
        await db.commit()
        cid = ch.id
    r = await client.put(f"/channels/{cid}/instagram/inbox", headers=h, json={"dm": True, "comments": True})
    assert r.status_code == 200 and r.json()["config"]["dm_reply"] is True
    async with SessionLocal() as db:
        bot = await ig_chat.bot_for(db, await db.get(Channel, cid))
        bot_id = bot.id

    old = _dm("m0", "قدیمی", minutes=-60)  # before answering was turned on: ignored
    shared = _dm("m2", "", minutes=2, item_type="media_share", media_share=SimpleNamespace(pk="3100", caption_text=""))
    fake.direct_threads = lambda *a, **k: [_thread("340282366841710300949128000000000001", [old, _dm("m1", "سلام قیمت؟"), shared])]
    req = _thread("340282366841710300949128000000000002", [_dm("r1", "موجوده؟", user_id="901")], user_pk="901",
                  username="newbie", pending=True)
    fake.direct_pending_inbox = lambda *a, **k: [req]
    approved = []
    fake.direct_pending_approve = lambda tid: approved.append(tid) or True
    fake.media_comments = lambda media, amount=20: [SimpleNamespace(
        pk="77", text="قیمت؟", user=SimpleNamespace(pk="902", username="commenter", full_name=""),
        created_at_utc=utcnow() + timedelta(minutes=1), replied_to_comment_id=None)]
    public = []
    fake.media_comment = lambda media, text, replied_to_comment_id=None: public.append((media, text, replied_to_comment_id))

    conv_ids = await ig_chat.poll_once(bot_id, comments_due=True)
    assert len(conv_ids) == 3
    assert approved == [340282366841710300949128000000000002]
    assert public == [("3100", "@commenter " + ig_chat.DEFAULT_COMMENT_REPLY, 77)]
    async with SessionLocal() as db:
        texts = (await db.execute(select(ChatMessage.text).where(ChatMessage.conversation_id.in_(conv_ids))
                                  .order_by(ChatMessage.id))).scalars().all()
        assert "سلام قیمت؟" in texts and "قدیمی" not in texts
        assert any("محصول «کیف چرم» (کد K7)" in t for t in texts)  # the shared post and the comment know the product
        convs = (await db.execute(select(ChatConversation).where(ChatConversation.id.in_(conv_ids)))).scalars().all()
        by_user = {c.customer_username: c for c in convs}
        assert by_user["commenter"].external_chat_id == "u:902"
    # A second round reads the same messages again: nothing new.
    assert await ig_chat.poll_once(bot_id, comments_due=True) == []

    # The assistant answers the commenter: first DM by user id, then the thread it created.
    sent = []
    fake.direct_send = lambda text, user_ids=[], thread_ids=[]: sent.append((text, user_ids, thread_ids)) or SimpleNamespace(
        id=f"s{len(sent)}", thread_id=340282366841710300949128000000000003)
    async with SessionLocal() as db:
        bot = await db.get(type(bot), bot_id)
        conv = await db.get(ChatConversation, by_user["commenter"].id)
        rows = [SimpleNamespace(external_id="")]
        await ig_chat.deliver(db, bot, conv, [SimpleNamespace(text="۱۲۰ هزار تومان", image_url="", buttons=[["ثبت سفارش"]])], rows)
        assert sent[0][1] == [902] and "• ثبت سفارش" in sent[0][0]
        assert conv.external_chat_id == "340282366841710300949128000000000003" and rows[0].external_id == "s1"
        assert "s1" in (await db.get(Channel, cid)).config["sent_ids"]
        await db.commit()

    # The seller typed in the Instagram app: the assistant steps back in that chat.
    fake.direct_threads = lambda *a, **k: [_thread("340282366841710300949128000000000001",
                                                   [_dm("m1", "سلام قیمت؟"), _dm("me1", "الان می‌فرستم", user_id="555", minutes=3)])]
    fake.direct_pending_inbox = lambda *a, **k: []
    assert await ig_chat.poll_once(bot_id, comments_due=False) == []
    async with SessionLocal() as db:
        conv = (await db.execute(select(ChatConversation).where(
            ChatConversation.external_chat_id == "340282366841710300949128000000000001"))).scalar_one()
        assert conv.status == "human"
