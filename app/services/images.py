"""Loading product images and normalising them to JPEG (Digikala and Instagram accept JPEG only)."""

import io
import uuid

import httpx

from app.platforms.base import EDIT_PRODUCT, PlatformError
from app.services.media import public_url, put, read_local

MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def load_bytes(url: str) -> bytes:
    if url.startswith("/media/"):
        if not (local := read_local(url)):
            raise PlatformError("فایل عکس محصول پیدا نشد.", "image_missing", EDIT_PRODUCT)
        data = local[0]
    else:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.content
        except httpx.HTTPError as exc:
            raise PlatformError("دانلود عکس محصول ناموفق بود.", "image_download", EDIT_PRODUCT, str(exc)) from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise PlatformError("حجم یکی از عکس‌ها بیش از ۱۰ مگابایت است.", "image_too_large", EDIT_PRODUCT)
    return data


def to_jpeg(data: bytes, max_side: int = 0) -> bytes:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=92)
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise PlatformError("یکی از عکس‌ها قابل تبدیل به JPEG نبود.", "bad_image", EDIT_PRODUCT, str(exc)) from exc


async def jpeg_bytes(url: str) -> tuple[bytes, str]:
    name = (url.rsplit("/", 1)[-1].split("?")[0].rsplit(".", 1)[0] or "photo") + ".jpg"
    return to_jpeg(await load_bytes(url)), name


async def public_jpeg(url: str) -> str:
    """Re-encode to JPEG under /media/out/ and return its public URL (Instagram fetches images itself)."""
    data, _ = await jpeg_bytes(url)
    return public_url(put(f"out/{uuid.uuid4().hex}.jpg", data, "image/jpeg"))
