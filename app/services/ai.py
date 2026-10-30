"""LLM layer. Keys, models and per-task routing come from the admin panel (ai_providers,
ai_task_routes); every call is written to ai_usage_logs for the monitoring page.

Wire formats: `gemini` (a key straight from Google AI Studio), `metis` (the same Gemini format through
the MetisAI gateway), `openai` (OpenAI and compatible gateways), `anthropic`. Each key may carry its own
proxy, stored with the key in the encrypted blob, because Google and OpenAI refuse requests from Iran."""

import asyncio
import base64
import json
import re
import time
from typing import Any

import httpx
from sqlalchemy import select

from app.core.db import SessionLocal, utcnow
from app.core.security import decrypt_credentials
from app.models import IMAGE_TASKS, AiProvider, AiTaskRoute, AiUsageLog

DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "metis": "https://api.metisai.ir",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


# ---------- model capabilities ----------
# Gemini lists TTS, audio, embedding and image models under generateContent too, so the method list alone
# can't tell what a model does. These name markers decide it; a wrong pick fails loudly before any call.
_UNSUPPORTED = (("tts", "فقط تبدیل متن به صدا است"), ("native-audio", "مدل صوتی است"), ("audio", "مدل صوتی است"),
                ("live", "مدل گفت‌وگوی زنده است"), ("embedding", "فقط بردار متن می‌سازد"), ("embed", "فقط بردار متن می‌سازد"),
                ("aqa", "فقط پاسخ به سؤال از متن است"), ("veo", "مدل ساخت ویدیو است"), ("lyria", "مدل ساخت موسیقی است"),
                ("imagen", "با این روش فراخوانی پشتیبانی نمی‌شود"), ("computer-use", "مدل کنترل کامپیوتر است"),
                ("robotics", "مدل رباتیک است"), ("transcribe", "فقط تبدیل صدا به متن است"),
                ("deep-research", "عامل تحقیق طولانی است، نه مدل پاسخ سریع"), ("antigravity", "عامل برنامه‌نویسی است"),
                ("translate", "فقط ترجمهٔ زنده است"))


def model_kind(model: str) -> tuple[str, str]:
    """('text'|'image'|'unsupported', reason_fa). Text models read photos and return JSON; image models return pictures."""
    m = (model or "").lower()
    for marker, reason in _UNSUPPORTED:
        if marker in m:
            return "unsupported", reason
    if "image" in m or "nano-banana" in m:
        return "image", "خروجی‌اش عکس است"
    return "text", ""


def check_model(task: str, model: str) -> str:
    """Empty when `model` fits `task`; otherwise a Persian reason for the admin."""
    if not model:
        return "مدل مشخص نشده است." if task in IMAGE_TASKS else ""
    kind, reason = model_kind(model)
    if kind == "unsupported":
        return f"مدل {model} برای Sello مناسب نیست؛ {reason}."
    if task in IMAGE_TASKS and kind != "image":
        return f"مدل {model} عکس تولید نمی‌کند؛ برای این کار یک مدل تصویری (نامش image دارد، مثل gemini-…-flash-image) انتخاب کن."
    if task not in IMAGE_TASKS and kind == "image":
        return f"مدل {model} مدل تصویری است و جواب متنی/JSON نمی‌دهد؛ یک مدل متنی مثل gemini-3.6-flash انتخاب کن."
    return ""


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


def provider_proxy(provider: AiProvider) -> str:
    return decrypt_credentials(provider.api_key_encrypted).get("proxy", "")


async def _call(
    provider: AiProvider, model: str, system: str, prompt: str, images: list[tuple[bytes, str]], temperature: float,
    json_mode: bool = True,
) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    key = _api_key(provider)
    base = (provider.base_url or DEFAULT_BASE_URLS[provider.kind]).rstrip("/")
    async with httpx.AsyncClient(timeout=120, proxy=provider_proxy(provider) or None) as client:
        if provider.kind in ("gemini", "metis"):
            parts: list[dict[str, Any]] = [{"text": prompt}]
            parts += [{"inlineData": {"mimeType": m, "data": base64.b64encode(d).decode()}} for d, m in images]
            resp = await client.post(
                f"{base}/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"role": "user", "parts": parts}],
                    "systemInstruction": {"parts": [{"text": system}]},
                    "generationConfig": {"temperature": temperature,
                                         **({"responseMimeType": "application/json"} if json_mode else {})},
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
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
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
            ] + [{"type": "text", "text": prompt + ("\n\nRespond with JSON only." if json_mode else "")}]
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


_BUSY = {"ai_http_503", "ai_http_500", "ai_http_502", "ai_http_504"}
BUSY_RETRY_DELAYS = (3.0, 8.0)  # Gemini answers 503 "high demand" in bursts; a short wait usually clears it


async def _retry_busy(fn):
    """Runs `fn` again on transient provider overload (5xx) before falling back to the next key/model."""
    for delay in (*BUSY_RETRY_DELAYS, None):
        try:
            return await fn()
        except AIError as exc:
            if exc.code not in _BUSY or delay is None:
                raise
            await asyncio.sleep(delay)


def _raise_for(resp: httpx.Response) -> None:
    low = resp.text[:2000].lower()
    if "location is not supported" in low or "unsupported_country" in low or "not available in your country" in low:
        raise AIError("این سرویس درخواست‌های ایران را قبول نمی‌کند؛ برای این کلید پروکسی خارج از ایران تنظیم کنید.",
                      code="ai_region_blocked", detail=resp.text[:500])
    if "api_key_invalid" in low or "api key not valid" in low or "incorrect api key" in low:
        raise AIError("کلید API نامعتبر است؛ کلید را دوباره کپی کنید یا یک کلید تازه بسازید.", code="ai_bad_key",
                      detail=resp.text[:500])
    if resp.status_code == 404 and ("model" in low or "not found" in low or "no longer available" in low):
        raise AIError("این مدل برای کلید شما در دسترس نیست؛ در ویرایش کلید، مدل پیش‌فرض را عوض کنید (مثلاً gemini-3.6-flash).",
                      code="ai_bad_model", detail=resp.text[:500])
    if resp.status_code in (401, 403):
        raise AIError("کلید API نامعتبر است یا دسترسی ندارد.", code=f"ai_http_{resp.status_code}", detail=resp.text[:500])
    if resp.status_code == 429 and "free_tier" in low and "limit: 0" in low:
        raise AIError("این مدل روی پلن رایگان گوگل سهمیه ندارد (limit 0)؛ برای استفاده باید Billing را روی پروژهٔ گوگلِ این کلید "
                      "فعال کنی یا کلید دیگری (پولی یا متیس) برای این کار بگذاری.", code="ai_free_tier", detail=resp.text[:500])
    if resp.status_code == 429:
        raise AIError("سقف استفاده از کلید API پر شده است.", code="ai_rate_limited", detail=resp.text[:500])
    if resp.status_code in (500, 502, 503, 504):
        raise AIError("سرویس هوش مصنوعی الان شلوغ است (خطای سمت گوگل/سرویس‌دهنده)؛ چند دقیقه بعد دوباره امتحان کن.",
                      code=f"ai_http_{resp.status_code}", detail=resp.text[:500])
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
        if not candidates and task not in IMAGE_TASKS:
            # No route for this task: use the first active key (its default model is always a text model).
            p = (await db.execute(select(AiProvider).where(AiProvider.is_active).order_by(AiProvider.id))).scalars().first()
            if p:
                candidates.append((p, p.default_model, temp, False))
        return [c for c in candidates if not check_model(task, c[1])
                and (task not in IMAGE_TASKS or c[0].kind in ("gemini", "metis"))]


async def route_problem(task: str) -> str:
    """Why `task` can't run with the current admin settings, in Persian; empty when it can."""
    async with SessionLocal() as db:
        route = (await db.execute(select(AiTaskRoute).where(AiTaskRoute.task == task))).scalar_one_or_none()
        if await _route(task):
            if task not in IMAGE_TASKS or not route:
                return ""
            # Don't keep calling an image model the provider already said has no quota on this key, until the admin
            # changes the route or the key (e.g. after enabling billing).
            last = (await db.execute(select(AiUsageLog).where(AiUsageLog.task == task)
                                     .order_by(AiUsageLog.id.desc()).limit(1))).scalar_one_or_none()
            provider = await db.get(AiProvider, route.provider_id) if route.provider_id else None
            changed = max(route.updated_at, provider.updated_at if provider else route.updated_at)
            no_quota = last and (last.error_code == "ai_free_tier"
                                 or ("free_tier" in last.error_message and "limit: 0" in last.error_message))
            if no_quota and _aware(last.created_at) > _aware(changed):
                return ("کلید فعلی گوگل روی پلن رایگان است و گوگل برای مدل‌های تصویری سهمیهٔ صفر می‌دهد. برای ساخت عکس، "
                        "Billing را روی پروژهٔ گوگلِ این کلید فعال کن یا در پنل ادمین کلید دیگری برای «تمیز کردن عکس» بگذار؛ "
                        "بعد از ذخیرهٔ دوبارهٔ کلید یا مسیر، دوباره امتحان می‌شود.")
            return ""
    if task in IMAGE_TASKS:
        if not route or not route.provider_id:
            return "برای «ساخت/تمیز کردن عکس» هنوز مدل تصویری تعیین نشده؛ ادمین در پنل ادمین ← هوش مصنوعی ← مسیر کارها یک کلید Gemini و یک مدل image انتخاب کند."
        return check_model(task, route.model) or "این کار فقط با کلید Gemini (مستقیم یا متیس) انجام می‌شود."
    if route and route.model and check_model(task, route.model):
        return check_model(task, route.model)
    return "هوش مصنوعی تنظیم نشده است؛ ادمین باید یک کلید API در پنل ادمین اضافه کند."


def _aware(dt):
    from datetime import timezone

    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def ai_available(task: str = "caption_extract") -> bool:
    return bool(await _route(task))


async def generate_json(
    *, task: str, system: str, prompt: str, images: list[tuple[bytes, str]] | None = None, store_id: int | None = None
) -> dict[str, Any]:
    candidates = await _route(task)
    if not candidates:
        raise AIError(await route_problem(task), code="ai_not_configured")
    last: AIError | None = None
    for provider, model, temp, is_fallback in candidates:
        started = time.monotonic()
        try:
            text, tin, tout = await _retry_busy(lambda: _call(provider, model, system, prompt, images or [], temp))
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


async def generate_text(
    *, task: str, system: str, prompt: str, images: list[tuple[bytes, str]] | None = None, store_id: int | None = None
) -> str:
    """Plain-text answer (e.g. an English image-edit prompt), with the same routing, fallback and logging."""
    candidates = await _route(task)
    if not candidates:
        raise AIError(await route_problem(task), code="ai_not_configured")
    last: AIError | None = None
    for provider, model, temp, is_fallback in candidates:
        started = time.monotonic()
        try:
            text, tin, tout = await _retry_busy(
                lambda: _call(provider, model, system, prompt, images or [], temp, json_mode=False))
            if not text.strip():
                raise AIError("مدل پاسخ خالی داد.", code="ai_empty")
            await _log(provider, task, model, store_id, ok=True, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000), input_tokens=tin, output_tokens=tout)
            return text.strip()
        except (AIError, httpx.HTTPError) as exc:
            err = exc if isinstance(exc, AIError) else AIError("ارتباط با سرویس هوش مصنوعی برقرار نشد.", code="ai_network",
                                                               detail=str(exc)[:500])
            await _log(provider, task, model, store_id, ok=False, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000),
                       error_code=err.code, error_message=f"{err.message_fa} {err.detail or ''}"[:1000])
            last = err
    raise last  # type: ignore[misc]


async def generate_image(
    *, system: str, prompt: str, images: list[tuple[bytes, str]], store_id: int | None = None, task: str = "image_edit",
) -> tuple[bytes, str]:
    """Image-to-image with a Gemini image model (same request shape platform-wrappers used):
    the raw photo goes in as inlineData, responseModalities IMAGE, square 2K output. Returns (bytes, mime)."""
    candidates = await _route(task)
    if not candidates:
        raise AIError(await route_problem(task), code="ai_not_configured")
    last: AIError | None = None
    for provider, model, temp, is_fallback in candidates:
        started = time.monotonic()
        base = (provider.base_url or DEFAULT_BASE_URLS[provider.kind]).rstrip("/")
        parts: list[dict[str, Any]] = [{"text": prompt}]
        parts += [{"inlineData": {"mimeType": m, "data": base64.b64encode(d).decode()}} for d, m in images]
        async def post(provider=provider, model=model, base=base, parts=parts, temp=temp):
            async with httpx.AsyncClient(timeout=180, proxy=provider_proxy(provider) or None) as client:
                resp = await client.post(
                    f"{base}/v1beta/models/{model}:generateContent", headers={"x-goog-api-key": _api_key(provider)},
                    json={"contents": [{"role": "user", "parts": parts}],
                          "systemInstruction": {"parts": [{"text": system}]},
                          "generationConfig": {"temperature": temp, "responseModalities": ["IMAGE"],
                                               "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"}}})
            _raise_for(resp)
            return resp.json()

        try:
            data = await _retry_busy(post)
            blob = next((p["inlineData"] for c in data.get("candidates", []) for p in c.get("content", {}).get("parts", [])
                         if isinstance(p.get("inlineData"), dict) and p["inlineData"].get("data")), None)
            if not blob:
                reason = (data.get("candidates") or [{}])[0].get("finishReason") or data.get("promptFeedback", {}).get("blockReason", "")
                raise AIError("عکس ساخته نشد؛ با یک عکس دیگر دوباره امتحان کن.", code="ai_no_image",
                              detail=json.dumps(data, ensure_ascii=False)[:500])
            usage = data.get("usageMetadata", {})
            await _log(provider, task, model, store_id, ok=True, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000),
                       input_tokens=usage.get("promptTokenCount", 0), output_tokens=usage.get("candidatesTokenCount", 0))
            return base64.b64decode(blob["data"]), blob.get("mimeType", "image/png")
        except (AIError, httpx.HTTPError) as exc:
            err = exc if isinstance(exc, AIError) else AIError("ارتباط با سرویس هوش مصنوعی برقرار نشد.", code="ai_network",
                                                               detail=str(exc)[:500])
            await _log(provider, task, model, store_id, ok=False, is_fallback=is_fallback,
                       latency_ms=int((time.monotonic() - started) * 1000),
                       error_code=err.code, error_message=f"{err.message_fa} {err.detail or ''}"[:1000])
            last = err
    raise last  # type: ignore[misc]


async def list_models(provider: AiProvider) -> list[str]:
    """Model ids the key can use for text generation, newest first as the provider lists them."""
    key = _api_key(provider)
    base = (provider.base_url or DEFAULT_BASE_URLS[provider.kind]).rstrip("/")
    async with httpx.AsyncClient(timeout=30, proxy=provider_proxy(provider) or None) as client:
        if provider.kind in ("gemini", "metis"):
            resp = await client.get(f"{base}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": key})
            _raise_for(resp)
            return [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                    and model_kind(m["name"].removeprefix("models/"))[0] != "unsupported"]
        if provider.kind == "openai":
            resp = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
            _raise_for(resp)
            return sorted((m["id"] for m in resp.json().get("data", [])), reverse=True)
        if provider.kind == "anthropic":
            resp = await client.get(f"{base}/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
            _raise_for(resp)
            return [m["id"] for m in resp.json().get("data", [])]
    return []


async def test_provider(provider: AiProvider, model: str | None = None) -> tuple[bool, str, int]:
    started = time.monotonic()
    if problem := check_model("product_generate", model or provider.default_model):
        return False, problem, 0
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
Title: what the product IS + its key distinguishing feature (type, material, colour, model), 40-90 characters,
no marketing words, no price. Description: a ready-to-publish Persian description of 4-8 short lines: what it is,
what it is made of / how it looks, who it suits and how it is used, then the facts as short bullet lines.
Facts (attributes, description bullets) must be grounded: only what is visible in the photos or stated by the seller
or the reference pages. If something (size, weight, material, brand) is not clear, omit it; never guess.
Brand only if it is written on the product or in the seller's note.
image_category: how the main photo should be staged: "flat_clothing" (garment without a person), "modeled_clothing"
(garment that must be shown worn: chador, abaya, manteau, dress), "footwear", or "general" (everything else).
Respond ONLY with JSON:
{"title":"...","description":"...","category_hint":"<Persian product type, e.g. ماگ سرامیکی>","brand":"","price":<int or 0>,
 "quantity":<int stock count the seller stated, else 0>,
 "attributes":{"<Persian name>":"value"},"tags":["..."],"image_category":"general",
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
