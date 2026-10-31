import httpx
import pytest

from app.core.security import encrypt_credentials
from app.models import AiProvider
from app.services import ai


def _provider(**kw):
    return AiProvider(id=1, name="g", kind=kw.get("kind", "gemini"), base_url="", default_model="gemini-2.5-flash",
                      api_key_encrypted=encrypt_credentials({"api_key": "AIza-test", "proxy": kw.get("proxy", "")}),
                      input_price_per_m=0, output_price_per_m=0)


def _patch(monkeypatch, handler, seen):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        seen["proxy"] = kw.pop("proxy", None)
        return real(*a, transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(ai.httpx, "AsyncClient", factory)


async def test_gemini_direct_uses_google_endpoint_key_header_and_proxy(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"], seen["key"] = str(req.url), req.headers.get("x-goog-api-key")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                                         "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}})

    _patch(monkeypatch, handler, seen)
    text, tin, tout = await ai._call(_provider(proxy="socks5://p:1080"), "gemini-2.5-flash", "s", "p", [], 0)
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert seen["key"] == "AIza-test" and seen["proxy"] == "socks5://p:1080"
    assert text == '{"ok": true}' and (tin, tout) == (5, 2)


async def test_gemini_region_block_is_explained(monkeypatch):
    def handler(req):
        return httpx.Response(400, json={"error": {"message": "User location is not supported for the API use."}})

    _patch(monkeypatch, handler, {})
    with pytest.raises(ai.AIError) as e:
        await ai._call(_provider(), "gemini-2.5-flash", "s", "p", [], 0)
    assert e.value.code == "ai_region_blocked"


async def test_metis_kind_uses_metis_gateway(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    _patch(monkeypatch, handler, seen)
    await ai._call(_provider(kind="metis"), "gemini-2.5-flash", "s", "p", [], 0)
    assert seen["url"].startswith("https://api.metisai.ir/v1beta/models/")


async def test_metis_routes_each_model_family_to_its_path(monkeypatch):
    seen = []

    def handler(req):
        seen.append((str(req.url), req.headers.get("authorization"), req.headers.get("x-api-key")))
        if "/anthropic/" in str(req.url):
            return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}], "usage": {}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    _patch(monkeypatch, handler, {})
    await ai._call(_provider(kind="metis"), "gpt-5-mini", "s", "p", [], 0)
    await ai._call(_provider(kind="metis"), "claude-sonnet-5-5", "s", "p", [], 0)
    await ai._call(_provider(kind="metis"), "deepseek-v4-flash", "s", "p", [], 0)
    assert seen[2][:2] == ("https://api.metisai.ir/deepseek/v1/chat/completions", "Bearer AIza-test")
    assert seen[0] == ("https://api.metisai.ir/openai/v1/chat/completions", "Bearer AIza-test", None)
    assert seen[1][0] == "https://api.metisai.ir/anthropic/v1/messages" and seen[1][2] == "AIza-test"


async def test_metis_lists_models_of_every_family(monkeypatch):
    def handler(req):
        u = str(req.url)
        if "/v1beta/models" in u:
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-3.1-flash-image", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]}]})
        if "/anthropic/v1/models" in u:
            return httpx.Response(200, json={"data": [{"id": "claude-sonnet-5-5"}]})
        if "/deepseek/v1/models" in u:
            return httpx.Response(200, json={"data": [{"id": "deepseek-v4-flash"}]})
        if "/openai/v1/models" in u:
            return httpx.Response(200, json={"data": [{"id": "gpt-5-mini"}, {"id": "tts-1"}, {"id": "text-embedding-3-large"},
                                                      {"id": "gpt-realtime"}, {"id": "gpt-image-1"}]})
        return httpx.Response(404)

    _patch(monkeypatch, handler, {})
    ids = await ai.list_models(_provider(kind="metis"))
    assert ids == ["gemini-3.6-flash", "gemini-3.1-flash-image", "claude-sonnet-5-5", "deepseek-v4-flash", "gpt-5-mini"]


async def test_busy_503_is_retried_before_failing(monkeypatch):
    hits = []

    def handler(req):
        hits.append(1)
        if len(hits) < 3:
            return httpx.Response(503, json={"error": {"message": "This model is currently experiencing high demand."}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]})

    _patch(monkeypatch, handler, {})
    monkeypatch.setattr(ai, "BUSY_RETRY_DELAYS", (0.0, 0.0))
    text, _, _ = await ai._retry_busy(lambda: ai._call(_provider(), "gemini-3.6-flash", "s", "p", [], 0))
    assert text == '{"ok": true}' and len(hits) == 3
