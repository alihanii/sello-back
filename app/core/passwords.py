"""Password hashing with scrypt (stdlib): memory-hard, salted, constant-time compare. Format:
scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>, so the cost can be raised later and old hashes still verify."""

import base64
import hashlib
import hmac
import secrets

N, R, P, DKLEN = 2**14, 8, 1, 32
MIN_LEN, MAX_LEN = 8, 128


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=DKLEN, maxmem=64 * 1024 * 1024)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        # Same work as a real check, so a missing password doesn't answer faster.
        hash_password(password or "x")
        return False
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        want = base64.b64decode(dk)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(want), maxmem=64 * 1024 * 1024)
    except Exception:
        return False
    return hmac.compare_digest(got, want)


def password_problem(password: str) -> str | None:
    """A reason the password is too weak, in Persian, or None."""
    if len(password) < MIN_LEN:
        return f"رمز باید دست‌کم {MIN_LEN} نویسه باشد."
    if len(password) > MAX_LEN:
        return "رمز خیلی طولانی است."
    if password.isdigit() and len(set(password)) <= 2:
        return "رمز خیلی ساده است."
    if password.lower() in {"password", "12345678", "123456789", "1234567890", "qwertyui", "11111111", "87654321"}:
        return "رمز خیلی ساده است."
    return None
