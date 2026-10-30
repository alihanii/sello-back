"""LLM layer. Keys, models and per-task routing come from the admin panel (ai_providers,
ai_task_routes); every call is written to ai_usage_logs for the monitoring page.

Wire formats: `gemini` (Google, or the MetisAI gateway platform-wrappers used), `openai`
(OpenAI and compatible gateways), `anthropic`."""

import base64
import json
import re
import time
from typing import Any

import httpx
from sqlalchemy import select

from app.core.db import SessionLocal, utcnow
from app.core.security import decrypt_credentials
from app.models import AiProvider, AiTaskRoute, AiUsageLog

DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


class AIError(Exception):
    def __init__(self, message_fa: str, code: str = "ai_error", detail: Any = None):
        super().__init__(message_fa)
        self.message_fa = message_fa
        self.code = code
        self.detail = detail


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text).strip()


def _api_key(provider: AiProvider) -> str:
    return decrypt_credentials(provider.api_key_encrypted).get("api_key", "")


async def _call(
    provider: AiProvider, model: str, system: str, prompt: str, images: list[tuple[bytes, str]], temperature: float
) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    key = _api_key(provider)
    base = (provider.base_url or DEFAULT_BASE_URLS[provider.kind]).rstrip("/")
    async with httpx.AsyncClient(timeout=120) as client:
        if provider.kind == "gemini":
            parts: list[dict[str, Any]] = [{"text": prompt}]
            parts += [{"inlineData": {"mimeType": m, "data": base64.b64encode(d).decode()}} for d, m in images]
            resp = await client.post(
                f"{base}/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"role": "user", "parts": parts}],
                    "systemInstruction": {"parts": [{"text": system}]},
                    "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
                },
            )
            _raise_for(resp)
            data = resp.json()
            text = "\n".join(
                p.get("text", "")
                for c in data.get("candidates", [])
                for p in c.get("content", {}).get("parts", [])
                if isinstance(p.get("text"), str)
            )
            usage = data.get("usageMetadata", {})
            return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

        if provider.kind == "openai":
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content += [
                {"type": "image_url", "image_url": {"url": f"data:{m};base64,{base64.b64encode(d).decode()}"}}
                for d, m in images
            ]
            resp = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                },
            )
            _raise_for(resp)
            data = resp.json()
            usage = data.get("usage", {})
            return (
                data["choices"][0]["message"]["content"] or "",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        if provider.kind == "anthropic":
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": m, "data": base64.b64encode(d).decode()}}
                for d, m in images
            ] + [{"type": "text", "text": prompt + "\n\nRespond with JSON only."}]
            resp = await client.post(
                f"{base}/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            _raise_for(resp)
            data = resp.json()
            usage = data.get("usage", {})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    raise AIError("نوع سرویس هوش مصنوعی پشتیبانی نمی‌شود.", code="ai_bad_kind")


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise AIError("کلید API نامعتبر است یا دسترسی ندارد.", code=f"ai_http_{resp.status_code}", detail=resp.text[:500])
    if resp.status_code == 429:
        raise AIError("سقف استفاده از کلید API پر شده است.", code="ai_rate_limited", detail=resp.text[:500])
    if resp.status_code >= 400:
        raise AIError("سرویس هوش مصنوعی خطا داد.", code=f"ai_http_{resp.status_code}", detail=resp.text[:500])


async def _log(provider: AiProvider, task: str, model: str, store_id: int | None, **fields: Any) -> None:
    tin, tout = fields.get("input_tokens", 0), fields.get("output_tokens", 0)
    cost = (tin * provider.input_price_per_m + tout * provider.output_price_per_m) // 1_000_000
    async with SessionLocal() as db:
        db.add(
            AiUsageLog(
                created_at=utcnow(), provider_id=provider.id, task=task, model=model, store_id=store_id,
                cost_toman=cost, **fields,
            )
        )
        await db.commit()


async def _route(task: str) -> list[tuple[AiProvider, str, float, bool]]:
    async with SessionLocal() as db:
        route = (await db.execute(select(AiTaskRoute).where(AiTaskRoute.task == task))).scalar_one_or_none()
        candidates: list[tuple[AiProvider, str, float, bool]] = []
        temp = (route.temperature if route else 30) / 100
        if route:
            for pid, model, fb in ((route.provider_id, route.model, False), (route.fallback_provider_id, route.fallback_model, True)):
                p = await db.get(AiProvider, pid) if pid else None
                if p and p.is_active:
                    candidates.append((p, model or p.default_model, temp, fb))
        if not candidates:
            # No route for this task: use the first active key.
            p = (await db.execute(select(AiProvider).where(AiProvider.is_active).order_by(AiProvider.id))).scalars().first()
            if p:
                candidates.append((p, p.default_model, temp, False))
        return candidates


async def ai_available(task: str = "caption_extract") -> bool:
    return bool(await _route(task))


async def generate_json(
    *, task: str, system: str, prompt: str, images: list[tuple[bytes, str]] | None = None, store_id: int | None = None
) -> dict[str, Any]:
    candidates = await _route(task)
    if not candidates:
        raise AIError("هوش مصنوعی تنظیم نشده است؛ ادمین باید یک کلید API در پنل ادمین اضافه کند.", code="ai_not_configured")
    last: AIError | None = None
    for provider, model, temp, is_fallback in candidates:
        started = time.monotonic()
        try:
            text, tin, tout = await _call(provider, model, system, prompt, images or [], temp)
            result = json.loads(_strip_fences(text))
            await _log(provider, task, model, store_id, ok=True, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000), input_tokens=tin, output_tokens=tout)
            return result
        except (AIError, httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            err = exc if isinstance(exc, AIError) else AIError(
                "پاسخ هوش مصنوعی قابل خواندن نبود." if isinstance(exc, (json.JSONDecodeError, KeyError))
                else "ارتباط با سرویس هوش مصنوعی برقرار نشد.",
                code="ai_bad_json" if isinstance(exc, (json.JSONDecodeError, KeyError)) else "ai_network",
                detail=str(exc)[:500],
            )
            await _log(provider, task, model, store_id, ok=False, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000),
                       error_code=err.code, error_message=f"{err.message_fa} {err.detail or ''}"[:1000])
            last = err
    raise last  # type: ignore[misc]


async def test_provider(provider: AiProvider, model: str | None = None) -> tuple[bool, str, int]:
    started = time.monotonic()
    try:
        text, tin, tout = await _call(provider, model or provider.default_model,
                                      "Reply with JSON.", 'Return {"ok": true}', [], 0)
        json.loads(_strip_fences(text))
        ms = int((time.monotonic() - started) * 1000)
        await _log(provider, "key_test", model or provider.default_model, None, ok=True, latency_ms=ms,
                   input_tokens=tin, output_tokens=tout)
        return True, "کلید سالم است و پاسخ داد.", ms
    except Exception as exc:  # noqa: BLE001 - we report every failure to the admin
        msg = exc.message_fa if isinstance(exc, AIError) else f"خطا: {exc}"
        ms = int((time.monotonic() - started) * 1000)
        await _log(provider, "key_test", model or provider.default_model, None, ok=False, latency_ms=ms,
                   error_code=getattr(exc, "code", "error"), error_message=msg[:1000])
        return False, msg, ms


# ---------- prompts ----------

CAPTION_EXTRACT_SYSTEM = """You extract structured product fields from social post captions (usually Persian).
Respond ONLY with JSON: {"posts":[{"message_id":"<copy>","title":"<short title>","description":"<text without price/stock lines>","price":<int Toman, 0 if missing>,"quantity":<int, 0 if missing>,"attributes":{"<name>":"<value>"}}]}
Rules: one object per input post, preserve message_id; numbers are integers; parse Persian digits;
do NOT invent data that is not in the caption."""

PRODUCT_FROM_REFERENCE_SYSTEM = """You are the product copywriter of an Iranian online shop. Write in Persian.
You get: optional product photos, optional notes from the seller, optional data extracted from reference
product pages, and optional STYLE PROFILE of the shop (how its existing products are written).
Build ONE new product for THIS shop. Use reference pages only as a template for structure and attributes;
never copy their text verbatim and never copy their brand unless the seller says so.
Respond ONLY with JSON:
{"title":"...","description":"...","category_hint":"...","brand":"","price":<int or 0>,
 "attributes":{"name":"value"},"tags":["..."],
 "field_sources":{"title":"image|link|style|note|guess","description":"...","attributes":"...","price":"..."}}
Do not invent prices or technical specs you cannot see or read; leave price 0 and omit unknown attributes."""

POST_WRITER_SYSTEM = """You write social media posts (Telegram, Bale, Instagram) for an Iranian online shop, in Persian.
You get the product's data, optional photos, the shop's STYLE PROFILE and recent example posts, the seller's
brief (tone, goals, campaign) and the captions of the shop's last AI posts (do not repeat their openings).
Write ONE ready-to-publish post in the shop's own voice: a hook line, the key features, sizes/colours if known,
a call to action, and the shop's usual hashtags/emojis. Do not invent specs, discounts or prices; do not write the
price (the app appends it). Respond ONLY with JSON: {"caption":"...","hook":"<first line>","hashtags":["#..."]}"""

STYLE_PROFILE_SYSTEM = """Analyse how this Iranian shop writes its product posts. Respond ONLY with JSON:
{"title_pattern":"...","tone":"friendly|formal","description_length":"short|medium|long",
 "caption_structure":["title","features","price","sizes","shipping","cta"],
 "common_hashtags":["#..."],"common_emojis":["..."],"always_mentions":["..."],
 "price_ranges":{"<category>":[min,max]},"example_title":"..."}"""
