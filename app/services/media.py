import uuid
from pathlib import Path

from app.core.config import settings

ALLOWED = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


def save_bytes(store_id: int, data: bytes, filename: str = "", content_type: str = "") -> str:
    ext = ALLOWED.get(content_type) or (Path(filename).suffix.lower() if Path(filename).suffix else ".jpg")
    folder = Path(settings.media_dir) / str(store_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{ext}"
    (folder / name).write_bytes(data)
    return f"/media/{store_id}/{name}"


def read_local(url: str) -> tuple[bytes, str] | None:
    if not url.startswith("/media/"):
        return None
    path = Path(settings.media_dir) / url.removeprefix("/media/")
    if not path.is_file():
        return None
    mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix, "image/jpeg")
    return path.read_bytes(), mime
