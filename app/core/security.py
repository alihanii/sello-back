import base64
import hashlib
import hmac
import json
from datetime import timedelta
from typing import Any

import jwt
from cryptography.fernet import Fernet

from app.core.config import settings
from app.core.db import utcnow

def _fernet_key() -> bytes:
    if settings.credentials_key:
        return settings.credentials_key.encode()
    # Dev fallback: stable across restarts so stored channel credentials stay readable.
    return base64.urlsafe_b64encode(hashlib.sha256(settings.jwt_secret.encode()).digest())


_fernet = Fernet(_fernet_key())


def create_access_token(user_id: int, role: str) -> str:
    payload = {"sub": str(user_id), "role": role, "exp": utcnow() + timedelta(minutes=settings.jwt_access_minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


def hash_code(phone: str, code: str) -> str:
    return hmac.new(settings.jwt_secret.encode(), f"{phone}:{code}".encode(), hashlib.sha256).hexdigest()


def encrypt_credentials(data: dict[str, Any]) -> str:
    return _fernet.encrypt(json.dumps(data).encode()).decode()


def decrypt_credentials(token: str | None) -> dict[str, Any]:
    if not token:
        return {}
    return json.loads(_fernet.decrypt(token.encode()))


SECRET_KEYS = {"token", "access_token", "bot_token", "password", "authorization", "api_key", "secret"}


def redact(value: Any) -> Any:
    """Strip secrets before a request/response is written to a log the user can see."""
    if isinstance(value, dict):
        return {k: ("***" if k.lower() in SECRET_KEYS else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and "/bot" in value and ":" in value:
        # Telegram/Bale URLs embed the bot token in the path.
        head, _, tail = value.partition("/bot")
        return head + "/bot***/" + tail.split("/", 1)[-1] if "/" in tail else head + "/bot***"
    return value
