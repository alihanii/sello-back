"""LLM layer. Keys, models and per-task routing come from the admin panel (ai_providers,
ai_task_routes); every call is written to ai_usage_logs for the monitoring page.

Wire formats: `gemini` (a key straight from Google AI Studio), `metis` (one MetisAI key for every model it
sells: Gemini models go out in the Gemini format, Claude through /anthropic, DeepSeek through /deepseek/v1, the rest through /openai/v1), `openai` (OpenAI and compatible gateways), `openrouter` (OpenAI format, plus image output),
`anthropic`. Each key may carry its own
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
from app.models import IMAGE_KINDS, IMAGE_TASKS, AiProvider, AiTaskRoute, AiUsageLog

DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "metis": "https://api.metisai.ir",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
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


def _base(provider: AiProvider) -> str:
    base = (provider.base_url or DEFAULT_BASE_URLS[provider.kind]).strip().rstrip("/")
    if provider.kind == "openrouter" and not base.endswith("/api/v1"):
        base = base.removesuffix("/api") + "/api/v1"  # "https://openrouter.ai" as typed from the site
    return base


_METIS_SUFFIXES = ("/openai/v1", "/deepseek/v1", "/anthropic/v1", "/anthropic", "/v1beta")


def metis_family(model: str) -> str:
    """Which MetisAI path serves a model: 'gemini', 'anthropic', 'deepseek' or 'openai'."""
    m = (model or "").lower().removeprefix("models/")
    if m.startswith(("gemini", "gemma", "nano-banana", "learnlm")):
        return "gemini"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _wire(provider: AiProvider, model: str) -> tuple[str, str]:
    """(wire format, base URL) for one call. Metis picks the format from the model family."""
    if provider.kind != "metis":
        return provider.kind, _base(provider)
    root = _base(provider)
    for suffix in _METIS_SUFFIXES:
        root = root.removesuffix(suffix)
    family = metis_family(model)
    # DeepSeek speaks the OpenAI chat format on its own path.
    return ("openai" if family == "deepseek" else family), {
        "gemini": root, "anthropic": f"{root}/anthropic", "openai": f"{root}/openai/v1", "deepseek": f"{root}/deepseek/v1"}[family]


def provider_proxy(provider: AiProvider) -> str:
    return decrypt_credentials(provider.api_key_encrypted).get("proxy", "")


async def _call(
    provider: AiProvider, model: str, system: str, prompt: str, images: list[tuple[bytes, str]], temperature: float,
    json_mode: bool = True, max_tokens: int | None = None, meta: dict[str, Any] | None = None,
) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens). `meta`, when given, receives `cached_tokens` (the part of the
    input the provider served from its prompt cache). `max_tokens` caps the answer (the chat assistant)."""
    key = _api_key(provider)
    kind, base = _wire(provider, model)
    meta = meta if meta is not None else {}
    async with httpx.AsyncClient(timeout=120, proxy=provider_proxy(provider) or None) as client:
        if kind == "gemini":
            parts: list[dict[str, Any]] = [{"text": prompt}]
            parts += [{"inlineData": {"mimeType": m, "data": base64.b64encode(d).decode()}} for d, m in images]
            gen: dict[str, Any] = {"temperature": temperature,
                                   **({"responseMimeType": "application/json"} if json_mode else {})}
            if max_tokens:
                gen["maxOutputTokens"] = max_tokens
                if "flash" in model.lower():
                    # Thinking tokens are billed as output and count against maxOutputTokens; short chat answers
                    # don't need them. Gemini 3 takes a level instead of a budget. A model (or a gateway like
                    # Metis) that rejects either gets the request again without it.
                    gen["thinkingConfig"] = ({"thinkingLevel": "minimal"} if re.match(r"gemini-([3-9]|\d\d)", model.lower())
                                             else {"thinkingBudget": 0})
            body = {"contents": [{"role": "user", "parts": parts}],
                    "systemInstruction": {"parts": [{"text": system}]}, "generationConfig": gen}
            url = f"{base}/v1beta/models/{model}:generateContent"
            resp = await client.post(url, headers={"x-goog-api-key": key}, json=body)
            if resp.status_code == 400 and "thinkingConfig" in gen:
                # Often a bare "Request contains an invalid argument." with no word about thinking.
                gen.pop("thinkingConfig")
                resp = await client.post(url, headers={"x-goog-api-key": key}, json=body)
            _raise_for(resp)
            data = resp.json()

            def _text(d: dict[str, Any]) -> str:
                return "\n".join(p.get("text", "") for c in d.get("candidates", [])
                                 for p in c.get("content", {}).get("parts", [])
                                 if isinstance(p.get("text"), str) and not p.get("thought"))

            text = _text(data)
            finish = ((data.get("candidates") or [{}])[0]).get("finishReason", "")
            if max_tokens and not text.strip() and finish == "MAX_TOKENS":
                # The model spent the whole cap thinking: once more with room to answer.
                gen["maxOutputTokens"] = max(4096, max_tokens * 4)
                resp = await client.post(url, headers={"x-goog-api-key": key}, json=body)
                _raise_for(resp)
                data = resp.json()
                text = _text(data)
            usage = data.get("usageMetadata", {})
            meta["cached_tokens"] = usage.get("cachedContentTokenCount", 0)
            return (text, usage.get("promptTokenCount", 0),
                    usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0))

        if kind in ("openai", "openrouter"):
            text = prompt + ("\n\nRespond with JSON only." if json_mode and kind == "openrouter" else "")
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            content += [
                {"type": "image_url", "image_url": {"url": f"data:{m};base64,{base64.b64encode(d).decode()}"}}
                for d, m in images
            ]
            req = {
                "model": model,
                "temperature": temperature,
                # Without a cap OpenRouter reserves the model's whole output (65536) and answers 402 on a small balance.
                **({"max_tokens": max_tokens or 8192} if kind == "openrouter" or max_tokens else {}),
                # OpenRouter drops every provider that lacks response_format (404); the prompt asks for JSON there.
                **({"response_format": {"type": "json_object"}} if json_mode and kind == "openai" else {}),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            }
            resp = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=req)
            _raise_for(resp)
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            if max_tokens and not ((choice.get("message") or {}).get("content") or "").strip() \
                    and choice.get("finish_reason") == "length":
                # A reasoning model used the whole cap before answering: once more with room to answer.
                req["max_tokens"] = max(4096, max_tokens * 4)
                resp = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=req)
                _raise_for(resp)
                data = resp.json()
            usage = data.get("usage", {})
            meta["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            return (
                data["choices"][0]["message"]["content"] or "",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        if kind == "anthropic":
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": m, "data": base64.b64encode(d).decode()}}
                for d, m in images
            ] + [{"type": "text", "text": prompt + ("\n\nRespond with JSON only." if json_mode else "")}]
            resp = await client.post(
                f"{base}/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": model,
                    "max_tokens": max_tokens or 4096,
                    "temperature": temperature,
                    # A cache breakpoint on the system prompt: Anthropic caches only when asked.
                    "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                    if max_tokens else system,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            _raise_for(resp)
            data = resp.json()
            usage = data.get("usage", {})
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            cached = usage.get("cache_read_input_tokens", 0) or 0
            meta["cached_tokens"] = cached
            tin = usage.get("input_tokens", 0) + cached + (usage.get("cache_creation_input_tokens", 0) or 0)
            return text, tin, usage.get("output_tokens", 0)

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


def _provider_message(resp: httpx.Response) -> str:
    """The provider's own error text (OpenAI/OpenRouter/Gemini put it in error.message), for the admin logs."""
    try:
        err = resp.json().get("error")
    except ValueError:
        return ""
    msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
    return " ".join(str(msg).split())[:300]


def _raise_for(resp: httpx.Response) -> None:
    low = resp.text[:2000].lower()
    said = _provider_message(resp)
    if resp.status_code == 404 and not resp.text.strip():
        # OpenRouter, Gemini and OpenAI always send a body with a 404; an empty one comes from a VPN or proxy on the way.
        raise AIError("پاسخ ۴۰۴ بدون متن آمد؛ این پاسخ از سرویس هوش مصنوعی نیست و معمولاً از VPN یا پروکسی سیستم "
                      "(مثلاً v2rayN) می‌آید. پروکسی سیستم را خاموش کن یا برای این کلید پروکسی درست بگذار.",
                      code="ai_proxy_404")
    if resp.status_code == 404 and "data policy" in low:
        raise AIError("OpenRouter این مدل را به‌خاطر تنظیمات حریم خصوصی حسابت اجرا نمی‌کند؛ در openrouter.ai/settings/privacy "
                      "گزینه‌های مدل‌های رایگان/آموزش را روشن کن یا یک مدل پولی انتخاب کن." + (f" ({said})" if said else ""),
                      code="ai_data_policy", detail=resp.text[:500])
    if resp.status_code == 404 and "no endpoints" in low:
        raise AIError("هیچ سرویس‌دهنده‌ای در OpenRouter این مدل را با این درخواست اجرا نمی‌کند؛ مدل را عوض کن."
                      + (f" ({said})" if said else ""), code="ai_bad_model", detail=resp.text[:500])
    if "location is not supported" in low or "unsupported_country" in low or "not available in your country" in low:
        raise AIError("این سرویس درخواست‌های ایران را قبول نمی‌کند؛ برای این کلید پروکسی خارج از ایران تنظیم کنید.",
                      code="ai_region_blocked", detail=resp.text[:500])
    if "api_key_invalid" in low or "api key not valid" in low or "incorrect api key" in low:
        raise AIError("کلید API نامعتبر است؛ کلید را دوباره کپی کنید یا یک کلید تازه بسازید.", code="ai_bad_key",
                      detail=resp.text[:500])
    if resp.status_code == 404 and ("model" in low or "not found" in low or "no longer available" in low):
        raise AIError("این مدل برای کلید شما در دسترس نیست؛ در ویرایش کلید، مدل پیش‌فرض را عوض کنید (مثلاً gemini-3.6-flash، یا در OpenRouter: google/gemini-3.6-flash).",
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
        raise AIError("سرویس هوش مصنوعی خطا داد." + (f" ({said})" if said else ""), code=f"ai_http_{resp.status_code}",
                      detail=resp.text[:500])


CACHED_INPUT_RATE = 0.25  # cached prompt tokens cost about a quarter of normal input on Gemini/OpenAI/Anthropic


def token_cost(provider: AiProvider, tin: int, tout: int, cached: int = 0) -> int:
    cached = min(cached, tin)
    return int(((tin - cached) * provider.input_price_per_m + cached * provider.input_price_per_m * CACHED_INPUT_RATE
                 + tout * provider.output_price_per_m) // 1_000_000)


async def _log(provider: AiProvider, task: str, model: str, store_id: int | None, **fields: Any) -> None:
    tin, tout = fields.get("input_tokens", 0), fields.get("output_tokens", 0)
    cost = token_cost(provider, tin, tout, fields.get("cached_tokens", 0))
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
                and (task not in IMAGE_TASKS or c[0].kind in IMAGE_KINDS)]


async def has_route(task: str) -> bool:
    async with SessionLocal() as db:
        return (await db.execute(select(AiTaskRoute.id).where(AiTaskRoute.task == task))).first() is not None


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


def parse_chat_json(text: str) -> dict[str, Any]:
    """The model's JSON, tolerating prose around it, and a cut-off answer (max_tokens hit mid-JSON): then the
    "reply" text alone is kept, so the customer still gets an answer instead of an error."""
    raw = _strip_fences(text)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    if m and m.group(1).strip():
        try:
            reply = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            reply = m.group(1)
        return {"reply": reply, "intent": "product", "partial": True}
    raise json.JSONDecodeError("no JSON object", text, 0)


CHAT_MAX_TEMPERATURE = 0.4  # sales answers must be steady; higher values drift into other languages


async def generate_chat(
    *, system: str, prompt: str, store_id: int | None, max_tokens: int = 400, task: str = "chat",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One chat turn: a JSON answer with a capped length. Returns (answer, usage) where usage has the model,
    input/output/cached tokens, cost and latency, so the conversation can show what each reply cost."""
    candidates = await _route(task)
    if not candidates:
        raise AIError(await route_problem(task), code="ai_not_configured")
    last: AIError | None = None
    for provider, model, temp, is_fallback in candidates:
        started = time.monotonic()
        meta: dict[str, Any] = {}
        try:
            temp = min(temp, CHAT_MAX_TEMPERATURE)
            tin = tout = 0
            for attempt in range(2):  # one quiet retry on unreadable output before the next key/model
                text, t_in, t_out = await _retry_busy(
                    lambda provider=provider, model=model, temp=temp: _call(
                        provider, model, system, prompt, [], temp, json_mode=True, max_tokens=max_tokens, meta=meta))
                tin, tout = tin + t_in, tout + t_out
                try:
                    result = parse_chat_json(text)
                    break
                except json.JSONDecodeError:
                    if attempt:
                        raise
                    temp = 0.0
            ms = int((time.monotonic() - started) * 1000)
            cached = int(meta.get("cached_tokens") or 0)
            await _log(provider, task, model, store_id, ok=True, is_fallback=is_fallback, latency_ms=ms,
                       input_tokens=tin, output_tokens=tout, cached_tokens=cached)
            return result, {"model": model, "input_tokens": tin, "output_tokens": tout, "cached_tokens": cached,
                            "cost_toman": token_cost(provider, tin, tout, cached), "latency_ms": ms}
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
        base = _wire(provider, model)[1]
        if provider.kind == "openrouter":
            try:
                data, mime, tin, tout = await _retry_busy(
                    lambda provider=provider, model=model, base=base, temp=temp:
                    _openrouter_image(provider, model, base, system, prompt, images, temp))
                await _log(provider, task, model, store_id, ok=True, is_fallback=is_fallback,
                           latency_ms=int((time.monotonic() - started) * 1000), input_tokens=tin, output_tokens=tout)
                return data, mime
            except (AIError, httpx.HTTPError) as exc:
                err = exc if isinstance(exc, AIError) else AIError("ارتباط با سرویس هوش مصنوعی برقرار نشد.",
                                                                   code="ai_network", detail=str(exc)[:500])
                await _log(provider, task, model, store_id, ok=False, is_fallback=is_fallback,
                           latency_ms=int((time.monotonic() - started) * 1000),
                           error_code=err.code, error_message=f"{err.message_fa} {err.detail or ''}"[:1000])
                last = err
                continue
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


async def _openrouter_image(provider: AiProvider, model: str, base: str, system: str, prompt: str,
                            images: list[tuple[bytes, str]], temp: float) -> tuple[bytes, str, int, int]:
    """OpenRouter image output: chat/completions with modalities, the picture comes back as a data URL."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": f"data:{m};base64,{base64.b64encode(d).decode()}"}}
                for d, m in images]
    async with httpx.AsyncClient(timeout=180, proxy=provider_proxy(provider) or None) as client:
        resp = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {_api_key(provider)}"},
                                 json={"model": model, "temperature": temp, "modalities": ["image", "text"],
                                       "image_config": {"aspect_ratio": "1:1"},
                                       "messages": [{"role": "system", "content": system},
                                                    {"role": "user", "content": content}]})
    _raise_for(resp)
    data = resp.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    url = next((i.get("image_url", {}).get("url", "") for i in msg.get("images") or []
                if i.get("image_url", {}).get("url", "").startswith("data:")), "")
    if not url:
        raise AIError("عکس ساخته نشد؛ با یک عکس دیگر دوباره امتحان کن.", code="ai_no_image",
                      detail=json.dumps(data, ensure_ascii=False)[:500])
    head, b64 = url.split(",", 1)
    usage = data.get("usage", {})
    return (base64.b64decode(b64), head.removeprefix("data:").split(";")[0] or "image/png",
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


async def list_models(provider: AiProvider) -> list[str]:
    """Model ids the key can use, newest first as the provider lists them. A Metis key lists every family it sells."""
    key = _api_key(provider)
    async with httpx.AsyncClient(timeout=30, proxy=provider_proxy(provider) or None) as client:
        async def fetch(kind: str, base: str) -> list[str]:
            if kind == "gemini":
                resp = await client.get(f"{base}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": key})
                _raise_for(resp)
                return [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                        and model_kind(m["name"].removeprefix("models/"))[0] != "unsupported"]
            if kind in ("openai", "openrouter"):
                resp = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                _raise_for(resp)
                ids = sorted((m["id"] for m in resp.json().get("data", [])), reverse=True)
                return [i for i in ids if not _not_chat(i)] if provider.kind == "metis" else ids
            if kind == "anthropic":
                resp = await client.get(f"{base}/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                _raise_for(resp)
                return [m["id"] for m in resp.json().get("data", [])]
            return []

        if provider.kind != "metis":
            return await fetch(provider.kind, _base(provider))
        samples = ("gemini", "claude", "deepseek", "gpt")
        results = await asyncio.gather(*(fetch(*_wire(provider, sample)) for sample in samples), return_exceptions=True)
        ok = [r for r in results if isinstance(r, list)]
        if not ok:
            raise next(r for r in results if isinstance(r, BaseException))
        return [m for r in ok for m in r]


# OpenAI lists speech, realtime, embedding, moderation and legacy completion models too; none answer chat JSON.
_NOT_CHAT = ("tts", "whisper", "transcribe", "realtime", "audio", "embedding", "moderation", "dall-e", "gpt-image",
             "babbage", "davinci", "search", "computer-use", "codex", "sora", "instruct")


def _not_chat(model: str) -> bool:
    m = model.lower()
    return any(x in m for x in _NOT_CHAT)


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
                   error_code=getattr(exc, "code", "error"),
                   error_message=f"{msg} {getattr(exc, 'detail', None) or ''}"[:1000])
        return False, msg, ms


# ---------- prompts ----------

CAPTION_EXTRACT_SYSTEM = """You extract structured product fields from social post captions (usually Persian).
Respond ONLY with JSON: {"posts":[{"message_id":"<copy>","title":"<short title>","description":"<text without price/stock lines>","price":<int Toman, 0 if missing>,"quantity":<int, 0 if missing>,"attributes":{"<name>":"<value>"}}]}
Rules: one object per input post, preserve message_id; numbers are integers; parse Persian digits;
do NOT invent data that is not in the caption."""

# "AI only" crawl mode: the same fields plus the product category and a fuller set of details, like AI product creation.
CAPTION_EXTRACT_FULL_SYSTEM = """You extract structured product fields from social post captions (usually Persian).
Respond ONLY with JSON: {"posts":[{"message_id":"<copy>","title":"<short title>","description":"<text without price/stock lines>","price":<int Toman, 0 if missing>,"quantity":<int, 0 if missing>,"category_hint":"<Persian product type, e.g. ماگ سرامیکی, هودی, کیف دوشی>","attributes":{"<Persian name>":"<value>"}}]}
Rules: one object per input post, preserve message_id; numbers are integers; parse Persian digits:
«۴ میلیون تومان» = 4000000, «۲۵۰ هزار» = 250000, «۲۰ تا» = 20.
category_hint: what the product IS, in a few Persian words, from the caption; empty if the caption doesn't say.
attributes: every concrete detail the caption states (جنس, رنگ, سایز, ابعاد, وزن, مدل, کشور سازنده, گارانتی …), one key per detail,
several values joined with «، ». Only what the caption says: do NOT invent data that is not in the caption."""

PRODUCT_FROM_PHOTO_SYSTEM = """You are the product copywriter of an Iranian online shop. Write in Persian.
You get: optional product photos and an optional short note from the seller.
Build ONE product for THIS shop.
Title: what the product IS + its key distinguishing feature (type, material, colour, model), 40-90 characters,
no marketing words, no price. Description: a ready-to-publish Persian description of 4-8 short lines: what it is,
what it is made of / how it looks, who it suits and how it is used, then the facts as short bullet lines.
Facts (attributes, description bullets) must be grounded: only what is visible in the photos or stated by the seller. If something (size, weight, material, brand) is not clear, omit it; never guess.
Brand only if it is written on the product or in the seller's note.
image_category: how the main photo should be staged: "flat_clothing" (garment without a person), "modeled_clothing"
(garment that must be shown worn: chador, abaya, manteau, dress), "footwear", or "general" (everything else).
Respond ONLY with JSON:
{"title":"...","description":"...","category_hint":"<Persian product type, e.g. ماگ سرامیکی>","brand":"","price":<int or 0>,
 "quantity":<int stock count the seller stated, else 0>,
 "attributes":{"<Persian name>":"value"},"tags":["..."],"image_category":"general",
 "field_sources":{"title":"image|note|guess","description":"...","attributes":"...","price":"..."}}
Price and quantity are plain integers in Toman: «۴ میلیون تومان» = 4000000, «۲۵۰ هزار» = 250000, «۲۰ تا» = 20.
Do not invent prices or technical specs you cannot see or read; leave price 0 and omit unknown attributes."""


