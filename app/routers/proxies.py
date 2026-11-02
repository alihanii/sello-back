"""Admin panel › proxies: the v2ray / socks / http list for Telegram and Instagram traffic, per-service mode
(auto, proxy only, direct) and live health. Links are secrets (they carry the server's uuid or password), so
they are stored encrypted and the panel only sees protocol, address and name."""

import asyncio
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.security import decrypt_credentials, encrypt_credentials
from app.deps import DB, AdminUser
from app.models import AppSetting, ProxyNode
from app.routers.admin import audit
from app.services.proxy_links import LinkError, parse_link, split_links
from app.services.proxy_pool import MODES, MODES_KEY, PURPOSES, pool

router = APIRouter(prefix="/admin/proxies", tags=["admin"])
MAX_NODES = 300


def _purposes(values: list[str]) -> list[str]:
    out = [p for p in dict.fromkeys(values) if p in PURPOSES]
    if not out:
        raise HTTPException(422, "حداقل یک کاربرد (تلگرام یا اینستاگرام) انتخاب کن.")
    return out


async def _state(db) -> dict:
    rows = (await db.execute(select(ProxyNode).order_by(ProxyNode.id))).scalars().all()
    active = {p: pool.active(p) for p in PURPOSES}
    nodes = []
    for r in rows:
        live = pool.nodes.get(r.id)
        nodes.append({
            "id": r.id, "name": r.name, "protocol": r.protocol, "address": r.address, "purposes": r.purposes or [],
            "enabled": r.enabled, "error": live.error if live else "",
            "health": {**(r.health or {}), **(live.health if live else {})},
            "active_for": [p for p, nid in active.items() if nid == r.id],
        })
    return {
        "nodes": nodes,
        "purposes": [{"id": k, "label": v["label"], "mode": pool.modes.get(k, "auto"),
                      "healthy": sum(1 for n in nodes if n["enabled"] and k in n["purposes"] and (n["health"].get(k) or {}).get("ok")),
                      "total": sum(1 for n in nodes if n["enabled"] and k in n["purposes"])} for k, v in PURPOSES.items()],
        "xray": {"installed": bool(pool.xray_binary()), "running": bool(pool.xray and pool.xray.returncode is None),
                 "error": pool.xray_error},
    }


@router.get("")
async def list_proxies(admin: AdminUser, db: DB):
    if not pool.nodes and await db.scalar(select(ProxyNode.id).limit(1)):
        await pool.reload()  # API process without the worker loop
    return await _state(db)


class ImportIn(BaseModel):
    text: str = ""
    subscription_url: str = ""
    purposes: list[str] = Field(default_factory=lambda: list(PURPOSES))


@router.post("/import")
async def import_proxies(body: ImportIn, admin: AdminUser, db: DB):
    purposes = _purposes(body.purposes)
    text = body.text
    if body.subscription_url.strip():
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(body.subscription_url.strip(), headers={"User-Agent": "v2rayN/7.0"})
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(422, f"لینک اشتراک دریافت نشد: {type(exc).__name__}") from exc
        text = text + "\n" + resp.text
    lines = split_links(text)
    if not lines:
        raise HTTPException(422, "هیچ لینکی پیدا نشد؛ هر لینک را در یک خط بگذار.")
    existing = {decrypt_credentials(r.link_encrypted).get("link", "")
                for r in (await db.execute(select(ProxyNode))).scalars()}
    if len(existing) + len(lines) > MAX_NODES:
        raise HTTPException(422, f"حداکثر {MAX_NODES} پروکسی می‌شود ذخیره کرد.")
    added, skipped, new_ids = 0, [], []
    for line in lines:
        if line in existing:
            skipped.append({"line": line[:60], "error": "تکراری است."})
            continue
        try:
            parsed = parse_link(line)
        except LinkError as exc:
            skipped.append({"line": line[:60], "error": str(exc)})
            continue
        row = ProxyNode(name=(parsed.name or parsed.address)[:120], protocol=parsed.protocol, address=parsed.address[:255],
                        link_encrypted=encrypt_credentials({"link": line}), purposes=purposes, enabled=True, health={})
        db.add(row)
        existing.add(line)
        added += 1
    await db.flush()
    new_ids = [r.id for r in (await db.execute(select(ProxyNode).order_by(ProxyNode.id.desc()).limit(added))).scalars()] if added else []
    audit(db, admin, "import_proxies", "proxy", None, {"added": added, "skipped": len(skipped), "purposes": purposes})
    await db.commit()
    await pool.reload()
    if new_ids:
        asyncio.create_task(pool.check_all(new_ids))
    return {"added": added, "skipped": skipped, **(await _state(db))}


class NodeIn(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    purposes: list[str] | None = None


@router.patch("/{node_id}")
async def update_proxy(node_id: int, body: NodeIn, admin: AdminUser, db: DB):
    row = await db.get(ProxyNode, node_id)
    if not row:
        raise HTTPException(404, "پروکسی پیدا نشد.")
    if body.name is not None:
        row.name = body.name.strip()[:120] or row.address
    if body.enabled is not None:
        row.enabled = body.enabled
    if body.purposes is not None:
        row.purposes = _purposes(body.purposes)
    audit(db, admin, "update_proxy", "proxy", node_id, body.model_dump(exclude_none=True))
    await db.commit()
    await pool.reload()
    return await _state(db)


@router.delete("/{node_id}")
async def delete_proxy(node_id: int, admin: AdminUser, db: DB):
    row = await db.get(ProxyNode, node_id)
    if not row:
        raise HTTPException(404, "پروکسی پیدا نشد.")
    await db.delete(row)
    audit(db, admin, "delete_proxy", "proxy", node_id, {"address": row.address})
    await db.commit()
    await pool.reload()
    return await _state(db)


@router.post("/delete-failed")
async def delete_failed(admin: AdminUser, db: DB):
    """Remove every node that failed its last check for all of its services."""
    removed = 0
    for row in (await db.execute(select(ProxyNode))).scalars().all():
        live = pool.nodes.get(row.id)
        health = {**(row.health or {}), **(live.health if live else {})}
        checks = [health.get(p) for p in row.purposes or []]
        if checks and all(c and c.get("ok") is False for c in checks):
            await db.delete(row)
            removed += 1
    audit(db, admin, "delete_failed_proxies", "proxy", None, {"removed": removed})
    await db.commit()
    await pool.reload()
    return {"removed": removed, **(await _state(db))}


@router.post("/check")
async def check_all(admin: AdminUser, db: DB):
    await pool.reload()
    await pool.check_all()
    return await _state(db)


@router.post("/{node_id}/check")
async def check_one(node_id: int, admin: AdminUser, db: DB):
    if node_id not in pool.nodes:
        await pool.reload()
    await pool.check_all([node_id])
    return await _state(db)


class ModesIn(BaseModel):
    modes: dict[str, Literal["auto", "proxy", "direct"]]


@router.put("/modes")
async def set_modes(body: ModesIn, admin: AdminUser, db: DB):
    row = await db.get(AppSetting, MODES_KEY)
    if not row:
        row = AppSetting(key=MODES_KEY, value={})
        db.add(row)
    row.value = {**(row.value or {}), **{k: v for k, v in body.modes.items() if k in PURPOSES and v in MODES}}
    audit(db, admin, "update_proxy_modes", "proxy", None, row.value)
    await db.commit()
    await pool.reload()
    return await _state(db)
