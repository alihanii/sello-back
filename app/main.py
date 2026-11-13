import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.db import init_db
from app.routers import (
    admin, auth, billing, channels, chat, chat_link, crawl, insights, notifications, ai_products, operations, orders, payments,
    products, proxies, site, stores,
)
from app.seed import seed
from app.services.chat_runtime import poller_loop
from app.services.ig_chat import poller_loop as ig_poller_loop
from app.services.li_chat import poller_loop as li_poller_loop
from app.services.dk_sync import poller_loop as dk_poller_loop
from app.services.media import public_url, use_s3
from app.services.proxy_pool import pool, pool_loop
from app.services.worker import worker_loop

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await seed()
    stop = asyncio.Event()
    task = asyncio.create_task(worker_loop(stop)) if settings.worker_enabled else None
    chat = (asyncio.create_task(poller_loop(stop))
            if settings.worker_enabled and settings.chat_transport == "polling" else None)
    proxies = asyncio.create_task(pool_loop(stop)) if settings.worker_enabled else None
    # Instagram has no webhooks for app-API accounts: their DMs and comments are read on a slow timer.
    ig_inbox = asyncio.create_task(ig_poller_loop(stop)) if settings.worker_enabled else None
    li_inbox = asyncio.create_task(li_poller_loop(stop)) if settings.worker_enabled else None  # LinkedIn: same, slower
    # Digikala: review state, price + stock after approval, and orders, on the admin's interval.
    dk_sync = asyncio.create_task(dk_poller_loop(stop)) if settings.worker_enabled else None
    yield
    stop.set()
    for t in (task, chat, proxies, ig_inbox, li_inbox, dk_sync):
        if t:
            await t
    await pool.shutdown()


app = FastAPI(title="Sello API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": {"message": "اطلاعات ارسالی کامل یا معتبر نیست.", "fields": fields}})


for r in (auth, stores, channels, products, operations, crawl, ai_products, insights, billing, payments, proxies, admin,
          notifications, chat, chat_link, orders, site):
    app.include_router(r.router, prefix="/api/v1")

if use_s3():
    @app.get("/media/{key:path}", include_in_schema=False)
    async def media_redirect(key: str):
        url = public_url(f"/media/{key}")
        if url.startswith("/media/"):
            raise HTTPException(404, "فایل پیدا نشد.")
        return RedirectResponse(url, status_code=302, headers={"Cache-Control": "public, max-age=300"})
else:
    Path(settings.media_dir).mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")


@app.get("/api/health")
async def health():
    return {"ok": True}
