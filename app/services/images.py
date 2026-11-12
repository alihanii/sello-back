"""Loading product images and normalising them to JPEG (Digikala and Instagram accept JPEG only)."""

import io
import uuid

import httpx

from app.platforms.base import EDIT_PRODUCT, PlatformError
from app.services.media import public_url, put, read_local

MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def load_bytes(url: str, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
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
    if len(data) > max_bytes:
        raise PlatformError("حجم یکی از فایل‌های محصول از حد مجاز بیشتر است.", "image_too_large", EDIT_PRODUCT)
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


STORY_SIZE = (1080, 1920)  # 9:16, Instagram's story frame


def story_frame(data: bytes) -> bytes:
    """A 1080x1920 (9:16) JPEG for a story: a photo that is already about 9:16 fills the frame; any other shape is
    shown whole in the middle over a blurred, darkened copy of itself, so nothing is cut and there are no bars."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        img = Image.open(io.BytesIO(to_jpeg(data)))
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = STORY_SIZE
        if abs(img.width / img.height - w / h) < 0.03:
            frame = ImageOps.fit(img, STORY_SIZE, Image.LANCZOS)
        else:
            frame = ImageOps.fit(img, (w // 6, h // 6), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6))
            frame = ImageEnhance.Brightness(frame.resize(STORY_SIZE, Image.LANCZOS)).enhance(0.55)
            fg = ImageOps.contain(img, (w, int(h * 0.82)), Image.LANCZOS)
            frame.paste(fg, ((w - fg.width) // 2, (h - fg.height) // 2))
        out = io.BytesIO()
        frame.save(out, "JPEG", quality=92)
        return out.getvalue()
    except PlatformError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PlatformError("عکس استوری آماده نشد.", "bad_image", EDIT_PRODUCT, str(exc)) from exc


async def jpeg_bytes(url: str) -> tuple[bytes, str]:
    name = (url.rsplit("/", 1)[-1].split("?")[0].rsplit(".", 1)[0] or "photo") + ".jpg"
    return to_jpeg(await load_bytes(url)), name


async def public_jpeg(url: str) -> str:
    """Re-encode to JPEG under /media/out/ and return its public URL (Instagram fetches images itself)."""
    data, _ = await jpeg_bytes(url)
    return public_url(put(f"out/{uuid.uuid4().hex}.jpg", data, "image/jpeg"))
