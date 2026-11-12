"""Product videos: checked with ffprobe on upload, and re-encoded with ffmpeg to the plain H.264/AAC MP4 Instagram
accepts (reels and video stories) right before publishing. ffmpeg is installed in the Docker image."""

import asyncio
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.platforms.base import EDIT_PRODUCT, PlatformError

MAX_VIDEO_BYTES = 50 * 1024 * 1024  # nginx allows 50 MB per request
MAX_SECONDS = 180  # Instagram reels made through the app API: up to 3 minutes
STORY_SECONDS = 60  # one story frame holds up to 60 seconds


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int


def available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


async def _run(*args: str, timeout: float = 300) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, b"", b"timeout"
    return proc.returncode or 0, out, err


async def probe(path: Path) -> VideoInfo:
    if not available():
        raise PlatformError("ffmpeg روی سرور نصب نیست؛ ویدیو قابل پردازش نیست.", "no_ffmpeg")
    code, out, _ = await _run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height:format=duration", "-of", "json", str(path), timeout=60)
    try:
        data = json.loads(out or b"{}")
        stream = (data.get("streams") or [{}])[0]
        info = VideoInfo(float(data["format"]["duration"]), int(stream["width"]), int(stream["height"]))
    except (KeyError, ValueError, TypeError, IndexError):
        info = None
    if code != 0 or info is None or info.duration <= 0:
        raise PlatformError("فایل ویدیو خوانده نشد؛ یک ویدیوی MP4 یا MOV سالم بفرست.", "bad_video", EDIT_PRODUCT)
    return info


async def probe_bytes(data: bytes, suffix: str) -> VideoInfo:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"in{suffix}"
        path.write_bytes(data)
        return await probe(path)


# A story frame is 1080x1920 (9:16): the video shown whole in the middle over a blurred copy of itself.
VERTICAL = ("[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2,eq=brightness=-0.15[b];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[f];"
            "[b][f]overlay=(W-w)/2:(H-h)/2,fps=30,setsar=1[v]")


async def for_instagram(src: Path, folder: Path, max_seconds: float, vertical: bool = False) -> tuple[Path, Path]:
    """(mp4, thumbnail jpg) in `folder`: H.264 + AAC, at most 1080 px wide, cut to `max_seconds`; `vertical` makes
    it a 1080x1920 story frame."""
    info = await probe(src)
    out, thumb = folder / "video.mp4", folder / "thumb.jpg"
    picture = (["-filter_complex", VERTICAL, "-map", "[v]", "-map", "0:a?"] if vertical
               else ["-vf", "scale='min(1080,iw)':-2,fps=30"])
    code, _, err = await _run(
        "ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_seconds),
        *picture, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart", str(out))
    if code != 0 or not out.is_file():
        raise PlatformError("آماده‌سازی ویدیو برای اینستاگرام ناموفق بود.", "video_encode", EDIT_PRODUCT,
                            err.decode("utf-8", "replace")[-300:])
    at = str(min(1.0, info.duration / 2))
    code, _, err = await _run("ffmpeg", "-y", "-v", "error", "-ss", at, "-i", str(out), "-frames:v", "1", "-q:v", "3",
                              str(thumb), timeout=60)
    if code != 0 or not thumb.is_file():
        raise PlatformError("ساخت کاور ویدیو ناموفق بود.", "video_thumb", EDIT_PRODUCT, err.decode("utf-8", "replace")[-300:])
    return out, thumb
