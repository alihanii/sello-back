"""Small in-memory sliding-window limiter (one backend process). Keys are e.g. "otp:<ip>"."""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.core.config import settings

_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    # nginx in front sets X-Real-IP / X-Forwarded-For; the left-most forwarded address is the client.
    fwd = request.headers.get("x-real-ip") or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "?")


def hit(key: str, limit: int, window_seconds: int) -> None:
    """Counts one request for `key`; raises 429 past `limit` in the window."""
    if not settings.ratelimit_enabled:
        return
    now = time.monotonic()
    q = _hits[key]
    while q and q[0] <= now - window_seconds:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "تعداد درخواست‌ها زیاد است؛ چند دقیقه بعد دوباره تلاش کنید.")
    q.append(now)
    if len(_hits) > 50_000:  # keep memory bounded
        for k in [k for k, v in _hits.items() if not v or v[-1] <= now - 3600][:25_000]:
            _hits.pop(k, None)


def reset() -> None:
    _hits.clear()
