# syntax=docker/dockerfile:1
FROM python:3.13-slim
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
