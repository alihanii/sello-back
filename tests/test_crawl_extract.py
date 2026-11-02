from app.platforms.base import CrawledItem
from app.services import ai, jobs


async def _ai_on(task):
    return True


async def test_ai_only_mode_adds_category_and_details(monkeypatch):
    seen = []

    async def fake(*, task, system, prompt, store_id=None, **kw):
        seen.append(system)
        return {"posts": [{"message_id": "1", "title": "هودی مشکی", "description": "هودی پنبه", "price": 850000,
                           "quantity": 3, "category_hint": "هودی", "attributes": {"جنس": "پنبه", "سایز": "L، XL"}}]}

    monkeypatch.setattr(ai, "ai_available", _ai_on)
    monkeypatch.setattr(ai, "generate_json", fake)
    item = CrawledItem(external_id="1", caption="هودی مشکی پنبه سایز L و XL قیمت ۸۵۰ هزار تومان")

    out, _ = await jobs._extract([item], "llm", 1)
    assert out["1"]["category_hint"] == "هودی" and out["1"]["attributes"] == {"جنس": "پنبه", "سایز": "L، XL"}
    assert seen[-1] is ai.CAPTION_EXTRACT_FULL_SYSTEM

    # "auto" keeps its current behaviour: the short prompt, no category
    out, _ = await jobs._extract([CrawledItem(external_id="1", caption="هودی")], "auto", 1)
    assert seen[-1] is ai.CAPTION_EXTRACT_SYSTEM and "category_hint" not in out["1"]


async def test_ai_only_mode_reads_each_post_with_its_photos(monkeypatch):
    """Telegram posts often put the details in the photo: AI only must see the photos and keep the attributes."""
    calls = []

    async def fake(*, task, system, prompt, images=None, store_id=None):
        import json
        posts = json.loads(prompt)["posts"]
        calls.append((posts, images))
        mid = posts[0]["message_id"]
        attrs = ({"رنگ": "مشکی", "جنس": "چرم"} if mid == "1"
                 else [{"name": "سایز", "value": ["۳۸", "۳۹"]}, {"name": "رنگ", "value": ""}])
        return {"posts": [{"message_id": mid, "title": f"محصول {mid}", "price": 0, "attributes": attrs}]}

    async def load(it):
        return [(f"img-{u}".encode(), "image/jpeg") for u in it.image_urls]

    monkeypatch.setattr(ai, "ai_available", _ai_on)
    monkeypatch.setattr(ai, "generate_json", fake)
    items = [CrawledItem(external_id="1", caption="کیف زنانه", image_urls=["a", "b", "c", "d"]),
             CrawledItem(external_id="2", caption="", image_urls=["e"])]

    out, notes = await jobs._extract(items, "llm", 1, load_images=load)
    assert not notes
    assert len(calls) == 2 and all(len(p) == 1 for p, _ in calls)  # one call per post
    imgs = {p[0]["message_id"]: im for p, im in calls}
    assert len(imgs["1"]) == jobs.AI_ONLY_PHOTOS and imgs["2"] == [(b"img-e", "image/jpeg")]
    assert out["1"]["attributes"] == {"رنگ": "مشکی", "جنس": "چرم"}
    assert out["2"]["attributes"] == {"سایز": "۳۸، ۳۹"} and out["2"]["title"] == "محصول 2"


def test_clean_attributes():
    assert jobs.clean_attributes(None) == {} and jobs.clean_attributes("x") == {}
    assert jobs.clean_attributes({"رنگ": None, " جنس ": "نخ", "وزن": 200}) == {"جنس": "نخ", "وزن": "200"}
