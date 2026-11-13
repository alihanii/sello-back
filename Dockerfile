# syntax=docker/dockerfile:1
FROM python:3.13-slim AS deps
WORKDIR /app

# Xray core for the smart proxies (admin panel › proxies): v2ray links run through it as local SOCKS ports.
ARG XRAY_VERSION=25.8.3
ARG TARGETARCH
RUN python - <<'PY'
import io, os, urllib.request, zipfile
arch = {"amd64": "64", "arm64": "arm64-v8a"}.get(os.environ.get("TARGETARCH") or "amd64", "64")
url = f"https://github.com/XTLS/Xray-core/releases/download/v{os.environ['XRAY_VERSION']}/Xray-linux-{arch}.zip"
data = urllib.request.urlopen(url, timeout=120).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    open("/usr/local/bin/xray", "wb").write(z.read("xray"))
os.chmod("/usr/local/bin/xray", 0o755)
PY
RUN xray version

# ffmpeg: product videos are re-encoded to the MP4 Instagram accepts (reels, video stories).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Everything above is the "deps" image (~300 MB): the deploy ships it to the server only when this file or
# requirements.txt changes. A usual deploy ships only app/ (a few hundred KB) and lays it on top there.
FROM deps
COPY app ./app
