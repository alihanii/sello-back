"""Uploaded and generated images.

Every image is stored under a key like "12/<uuid>.jpg" and referenced everywhere (database, API, frontend) by the
URL "/media/<key>". Where the bytes live is set by SELLO_STORAGE:
- "local" (default): files in SELLO_MEDIA_DIR, served by the /media route.
- "s3": an S3-compatible bucket (AWS, ArvanCloud, MinIO, ...); the /media route redirects to the bucket.
"""

import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from app.core.config import settings

ALLOWED = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}


def _mime(key: str) -> str:
    return MIME.get(Path(key).suffix.lower(), "image/jpeg")


def use_s3() -> bool:
    return settings.storage == "s3"


@lru_cache
def _s3():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        region_name=settings.s3_region or None,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(
            s3={"addressing_style": settings.s3_addressing_style},
            # S3-compatible providers often reject the newer default checksum headers.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            connect_timeout=10, read_timeout=60, retries={"max_attempts": 3},
        ),
    )


def _object(key: str) -> str:
    return f"{settings.s3_prefix.strip('/')}/{key}" if settings.s3_prefix.strip("/") else key


def _key(url: str) -> str | None:
    if not url.startswith("/media/"):
        return None
    key = url.removeprefix("/media/").split("?")[0]
    return None if not key or ".." in key.split("/") else key


def put(key: str, data: bytes, content_type: str = "") -> str:
    """Store bytes under key; returns the /media URL."""
    if use_s3():
        extra = {"ACL": settings.s3_acl} if settings.s3_acl else {}
        _s3().put_object(Bucket=settings.s3_bucket, Key=_object(key), Body=data,
                         ContentType=content_type or _mime(key), CacheControl="public, max-age=31536000", **extra)
    else:
        path = Path(settings.media_dir) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return f"/media/{key}"


def save_bytes(store_id: int, data: bytes, filename: str = "", content_type: str = "") -> str:
    ext = ALLOWED.get(content_type) or (Path(filename).suffix.lower() if Path(filename).suffix else ".jpg")
    return put(f"{store_id}/{uuid.uuid4().hex}{ext}", data, ALLOWED.get(content_type) and content_type or "")


def read_local(url: str) -> tuple[bytes, str] | None:
    """The bytes and mime type of one of our /media URLs; None for other URLs or a missing file."""
    key = _key(url)
    if key is None:
        return None
    if use_s3():
        try:
            obj = _s3().get_object(Bucket=settings.s3_bucket, Key=_object(key))
        except Exception:  # noqa: BLE001  (missing key, network)
            return None
        return obj["Body"].read(), obj.get("ContentType") or _mime(key)
    path = Path(settings.media_dir) / key
    if not path.is_file():
        return None
    return path.read_bytes(), _mime(key)


def public_url(url: str) -> str:
    """An absolute address anyone can download a /media URL from (Instagram fetches images itself)."""
    key = _key(url)
    if key is None:
        return url
    if use_s3():
        if settings.s3_public_url:
            return f"{settings.s3_public_url.rstrip('/')}/{quote(_object(key))}"
        return _s3().generate_presigned_url("get_object", Params={"Bucket": settings.s3_bucket, "Key": _object(key)},
                                            ExpiresIn=settings.s3_url_expire_seconds)
    return f"{settings.public_base_url.rstrip('/')}/media/{key}"
