import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.db import init_db
from app.routers import (
    admin, auth, automations, billing, channels, crawl, insights, notifications, operations, payments, products, references,
    stores,
)
from app.seed import seed
from app.services.worker import worker_loop

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await seed()
    stop = asyncio.Event()
    task = asyncio.create_task(worker_loop(stop)) if settings.worker_enabled else None
    yield
    stop.set()
    if task:
        await task


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


for r in (auth, stores, channels, products, operations, crawl, references, automations, insights, billing, payments, admin,
          notifications):
    app.include_router(r.router, prefix="/api/v1")

Path(settings.media_dir).mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")


@app.get("/api/health")
async def health():
    return {"ok": True}
