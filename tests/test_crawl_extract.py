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
