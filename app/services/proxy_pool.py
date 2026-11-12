"""Smart outbound proxies for Telegram and Instagram traffic.

The admin pastes v2ray links (or plain socks/http proxies) and says which service each one is for. v2ray links
run through one local Xray process: each node gets its own SOCKS port on 127.0.0.1. Every few minutes each node
is tested against the service it serves (api.telegram.org, graph.instagram.com) and its latency recorded.
`request()` sends a call through the fastest healthy node and, when the connection fails before anything was
sent, moves on to the next one, so a dead node never blocks a publish or a chat reply.

Per service the admin picks a mode: auto (pool, then the old single-proxy field, then direct), proxy (pool and
the old field only, never direct) or direct (no proxy at all)."""

import asyncio
import json
import logging
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal, utcnow
from app.core.security import decrypt_credentials
from app.models import AppSetting, ProxyNode
from app.services.proxy_links import LinkError, parse_link, xray_config

log = logging.getLogger("sello.proxy")

PURPOSES: dict[str, dict[str, str]] = {
    "telegram": {"label": "تلگرام", "test_url": "https://api.telegram.org/"},
    "instagram": {"label": "اینستاگرام", "test_url": "https://graph.instagram.com/"},
    "linkedin": {"label": "لینکدین", "test_url": "https://www.linkedin.com/"},
}
MODES = ("auto", "proxy", "direct")
MODES_KEY = "proxy_modes"
FAILOVER_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError)
CHECK_TIMEOUT = 12.0
CONNECT_TIMEOUT = 12.0  # through a proxy; a dead node must not hold a request for the whole read timeout
FAIL_COOLDOWN = 90.0  # a node that just failed is skipped this long (or until the next good check)


@dataclass
class _Node:
    id: int
    name: str
    protocol: str
    purposes: list[str]
    outbound: dict[str, Any] = field(default_factory=dict)
    proxy_url: str = ""  # socks/http nodes: used directly
    port: int = 0  # xray nodes: local socks port
    error: str = ""  # can't run at all (bad link, no xray)
    health: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_at: dict[str, float] = field(default_factory=dict)
    channel_id: int | None = None  # a seller's own proxy for one account; never shared

    @property
    def url(self) -> str:
        return self.proxy_url or (f"socks5://127.0.0.1:{self.port}" if self.port else "")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Pool:
    def __init__(self) -> None:
        self.nodes: dict[int, _Node] = {}
        self.modes: dict[str, str] = {p: "auto" for p in PURPOSES}
        self.ports: dict[int, int] = {}
        self.xray: asyncio.subprocess.Process | None = None
        self.xray_config = ""
        self.xray_error = ""
        self.recheck = asyncio.Event()
        self._lock = asyncio.Lock()

    # ---------- xray ----------

    @staticmethod
    def xray_binary() -> str:
        return shutil.which(settings.xray_path) or ""

    async def _apply_xray(self, nodes: list[_Node]) -> None:
        runnable = [n for n in nodes if n.outbound and not n.error]
        if not runnable:
            await self._stop_xray()
            self.xray_config, self.xray_error = "", ""
            return
        binary = self.xray_binary()
        if not binary:
            self.xray_error = "هستهٔ Xray روی سرور نصب نیست (SELLO_XRAY_PATH)؛ لینک‌های v2ray اجرا نمی‌شوند."
            for n in runnable:
                n.error, n.port = self.xray_error, 0
            return
        for n in runnable:
            n.port = self.ports.setdefault(n.id, _free_port())
        config = json.dumps(xray_config([(str(n.id), n.port, n.outbound) for n in runnable]), ensure_ascii=False, indent=1)
        if config == self.xray_config and self.xray and self.xray.returncode is None:
            return
        await self._stop_xray()
        folder = Path(settings.xray_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        path.write_text(config, encoding="utf-8")
        log_file = open(folder / "xray.log", "ab")  # noqa: SIM115 - kept open for the process
        try:
            self.xray = await asyncio.create_subprocess_exec(binary, "run", "-c", str(path), stdout=log_file, stderr=log_file)
        except OSError as exc:
            self.xray_error = f"اجرای Xray ناموفق بود: {exc}"
            log_file.close()
            return
        await asyncio.sleep(0.8)
        if self.xray.returncode is not None:
            tail = (folder / "xray.log").read_bytes()[-600:].decode("utf-8", "replace")
            self.xray_error = "Xray اجرا نشد: " + tail.strip().splitlines()[-1][:300] if tail.strip() else "Xray اجرا نشد."
            self.xray = None
            log.warning("xray exited: %s", tail)
            return
        self.xray_config, self.xray_error = config, ""
        log.info("xray running with %d nodes", len(runnable))

    async def _stop_xray(self) -> None:
        if self.xray and self.xray.returncode is None:
            self.xray.terminate()
            try:
                await asyncio.wait_for(self.xray.wait(), 5)
            except asyncio.TimeoutError:
                self.xray.kill()
        self.xray = None

    async def shutdown(self) -> None:
        await self._stop_xray()

    # ---------- state ----------

    async def reload(self) -> None:
        """Read nodes and modes from the database and (re)start Xray when its config changed."""
        async with self._lock:
            async with SessionLocal() as db:
                rows = (await db.execute(select(ProxyNode).where(ProxyNode.enabled.is_(True)).order_by(ProxyNode.id))).scalars().all()
                modes_row = await db.get(AppSetting, MODES_KEY)
            self.modes = {p: ((modes_row.value or {}).get(p) if modes_row else None) or "auto" for p in PURPOSES}
            nodes: list[_Node] = []
            for r in rows:
                old = self.nodes.get(r.id)
                n = _Node(r.id, r.name, r.protocol, list(r.purposes or []), health=dict(r.health or {}),
                          channel_id=r.channel_id)
                if old:
                    n.failed_at = old.failed_at
                    n.health = {**n.health, **old.health}
                try:
                    parsed = parse_link(decrypt_credentials(r.link_encrypted).get("link", ""))
                    n.outbound, n.proxy_url = parsed.outbound, parsed.proxy_url
                except LinkError as exc:
                    n.error = str(exc)
                nodes.append(n)
            await self._apply_xray(nodes)
            if self.xray is None or self.xray.returncode is not None:
                for n in nodes:
                    if n.outbound and not n.error:
                        n.error = self.xray_error or "Xray اجرا نیست."
            self.nodes = {n.id: n for n in nodes}
            self.ports = {k: v for k, v in self.ports.items() if k in self.nodes}

    def _usable(self, purpose: str) -> list[_Node]:
        return [n for n in self.nodes.values() if purpose in n.purposes and n.url and not n.error and n.channel_id is None]

    def candidates(self, purpose: str) -> list[_Node]:
        """Healthy nodes by latency, then not yet tested ones; nodes failed recently come last."""
        now = time.monotonic()

        def rank(n: _Node) -> tuple:
            h = n.health.get(purpose) or {}
            cooling = now - n.failed_at.get(purpose, -1e9) < FAIL_COOLDOWN
            if cooling or h.get("ok") is False:
                return (2, h.get("latency_ms") or 1e9)
            if h.get("ok"):
                return (0, h.get("latency_ms") or 1e9)
            return (1, n.id)

        ranked = sorted(self._usable(purpose), key=rank)
        good = [n for n in ranked if rank(n)[0] < 2]
        # When every node looks dead, still try the best two: health checks may simply be stale.
        return good or ranked[:2]

    def routes(self, purpose: str, fallback_proxy: str = "") -> list[tuple[int | None, str | None]]:
        mode = self.modes.get(purpose, "auto")
        if mode == "direct":
            return [(None, None)]
        out: list[tuple[int | None, str | None]] = [(n.id, n.url) for n in self.candidates(purpose)]
        if fallback_proxy:
            out.append((None, fallback_proxy))
        if mode == "auto" or not out:
            out.append((None, None))
        return out

    def sticky(self, purpose: str, preferred: int | None, load: dict[int, int] | None = None,
               exclude: tuple[int, ...] = (), channel_id: int | None = None) -> tuple[int | None, str] | None:
        """One fixed route for an account (Instagram app-API logins must keep one IP): the account's own node while
        it is healthy, else the healthy node fewest accounts use. Only the admin's nodes, never direct, unless the
        admin set the service to "direct". None when there is no usable node.
        An account with its own proxy (the seller's fixed IP) always uses that one and nothing else."""
        own = next((n for n in self.nodes.values() if channel_id is not None and n.channel_id == channel_id), None)
        if own:
            return (own.id, own.url) if own.url and not own.error else None
        if self.modes.get(purpose, "auto") == "direct":
            return (None, "")
        good = [n for n in self.candidates(purpose) if n.id not in exclude]
        now = time.monotonic()
        healthy = [n for n in good if (n.health.get(purpose) or {}).get("ok") is not False
                   and now - n.failed_at.get(purpose, -1e9) >= FAIL_COOLDOWN]
        for n in healthy:
            if n.id == preferred:
                return (n.id, n.url)
        pick = sorted(healthy or good, key=lambda n: (load or {}).get(n.id, 0))
        return (pick[0].id, pick[0].url) if pick else None

    def active(self, purpose: str) -> int | None:
        """The node requests for `purpose` use right now, if it passed its last check."""
        node_id, _ = self.routes(purpose)[0]
        n = self.nodes.get(node_id) if node_id is not None else None
        return node_id if n and (n.health.get(purpose) or {}).get("ok") else None

    def report(self, node_id: int, purpose: str, ok: bool, latency_ms: int | None = None, error: str = "") -> None:
        n = self.nodes.get(node_id)
        if not n:
            return
        if ok:
            n.failed_at.pop(purpose, None)
            prev = n.health.get(purpose) or {}
            n.health[purpose] = {**prev, "ok": True, "latency_ms": latency_ms or prev.get("latency_ms"), "error": "",
                                 "checked_at": utcnow().isoformat()}
        else:
            n.failed_at[purpose] = time.monotonic()
            n.health[purpose] = {"ok": False, "latency_ms": None, "error": error[:300], "checked_at": utcnow().isoformat()}
            self.recheck.set()

    # ---------- health ----------

    async def check(self, node_id: int, purpose: str) -> dict[str, Any]:
        n = self.nodes.get(node_id)
        if not n or n.error or not n.url:
            res = {"ok": False, "latency_ms": None, "error": (n.error if n else "") or "غیرفعال", "checked_at": utcnow().isoformat()}
            if n:
                n.health[purpose] = res
            return res
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=CHECK_TIMEOUT, proxy=n.url) as client:
                await client.get(PURPOSES[purpose]["test_url"])
        except httpx.HTTPError as exc:
            self.report(node_id, purpose, False, error=_short(exc))
        else:
            self.report(node_id, purpose, True, int((time.monotonic() - t0) * 1000))
        return n.health[purpose]

    async def check_all(self, only: list[int] | None = None) -> None:
        sem = asyncio.Semaphore(8)

        async def one(nid: int, purpose: str) -> None:
            async with sem:
                await self.check(nid, purpose)

        jobs = [one(n.id, p) for n in list(self.nodes.values()) if only is None or n.id in only
                for p in n.purposes if p in PURPOSES]
        await asyncio.gather(*jobs)
        await self.persist()

    async def persist(self) -> None:
        async with SessionLocal() as db:
            for n in self.nodes.values():
                row = await db.get(ProxyNode, n.id)
                if row:
                    row.health = dict(n.health)
            await db.commit()


def _short(exc: Exception) -> str:
    text = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:300]


pool = Pool()


async def request(purpose: str, method: str, url: str, *, fallback_proxy: str = "", timeout: float = 60,
                  follow_redirects: bool = False, **kw) -> httpx.Response:
    """One HTTP call for `purpose`, through the best route; fails over only when the connection itself failed
    (nothing reached the service yet), so a POST is never sent twice."""
    last: Exception | None = None
    for node_id, proxy in pool.routes(purpose, fallback_proxy):
        t = httpx.Timeout(timeout, connect=min(CONNECT_TIMEOUT, timeout)) if proxy else httpx.Timeout(timeout)
        try:
            async with httpx.AsyncClient(timeout=t, proxy=proxy, follow_redirects=follow_redirects) as client:
                resp = await client.request(method, url, **kw)
        except FAILOVER_ERRORS as exc:
            if node_id is not None:
                pool.report(node_id, purpose, False, error=_short(exc))
                log.info("proxy %s failed for %s, trying the next route: %s", node_id, purpose, exc)
            last = exc
            continue
        if node_id is not None:
            pool.report(node_id, purpose, True)  # latency comes from the checks; a big upload would skew it
        return resp
    assert last is not None
    raise last


def uses_proxy(purpose: str, fallback_proxy: str = "") -> bool:
    return any(proxy for _, proxy in pool.routes(purpose, fallback_proxy))


async def pool_loop(stop: asyncio.Event) -> None:
    """Re-test every node every `proxy_check_seconds`, and soon after a live request reported a failure."""
    try:
        await pool.reload()
        await pool.check_all()
    except Exception:  # noqa: BLE001
        log.exception("proxy pool start failed")
    while not stop.is_set():
        pool.recheck.clear()
        waiter = asyncio.ensure_future(pool.recheck.wait())
        stopper = asyncio.ensure_future(stop.wait())
        done, pending = await asyncio.wait({waiter, stopper}, timeout=settings.proxy_check_seconds,
                                           return_when=asyncio.FIRST_COMPLETED)
        for f in pending:
            f.cancel()
        if stop.is_set():
            break
        if waiter in done:  # a failure: give the node a moment, then check only the failed ones
            await asyncio.sleep(min(FAIL_COOLDOWN, 30))
            failed = [n.id for n in pool.nodes.values() if n.failed_at]
            try:
                await pool.check_all(failed)
            except Exception:  # noqa: BLE001
                log.exception("proxy recheck failed")
            continue
        try:
            if pool.xray is not None and pool.xray.returncode is not None:
                pool.xray_config = ""  # crashed: start it again
            await pool.reload()
            await pool.check_all()
        except Exception:  # noqa: BLE001
            log.exception("proxy check failed")
    await pool.shutdown()
