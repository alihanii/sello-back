"""Share links of the v2ray family, turned into Xray outbounds.

Supported: vless:// (tcp, ws, grpc, httpupgrade, xhttp, kcp; tls or reality), vmess:// (v2rayN base64 JSON),
trojan://, ss:// (SIP002 and the older all-base64 form), and plain socks5:// / socks:// / http:// / https://
proxies, which httpx uses directly without Xray."""

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

XRAY_PROTOCOLS = {"vless", "vmess", "trojan", "ss"}
DIRECT_PROTOCOLS = {"socks", "http"}


class LinkError(ValueError):
    """The link can't be used; `args[0]` is a Persian message for the admin."""


@dataclass
class ParsedLink:
    protocol: str  # vless | vmess | trojan | ss | socks | http
    name: str
    host: str
    port: int
    outbound: dict[str, Any] = field(default_factory=dict)  # Xray outbound (without tag); empty for direct ones
    proxy_url: str = ""  # for socks/http: the URL httpx uses as is

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def needs_xray(self) -> bool:
        return self.protocol in XRAY_PROTOCOLS


def _b64(text: str) -> str:
    text = text.strip().replace("-", "+").replace("_", "/")
    return base64.b64decode(text + "=" * (-len(text) % 4)).decode("utf-8")


def _port(value: Any) -> int:
    try:
        port = int(str(value).strip())
    except ValueError:
        raise LinkError("پورت لینک نامعتبر است.") from None
    if not 0 < port < 65536:
        raise LinkError("پورت لینک نامعتبر است.")
    return port


def _q(query: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(query, keep_blank_values=True).items()}


def _stream(p: dict[str, str], host: str) -> dict[str, Any]:
    """streamSettings from the common query keys: type, security, sni, host, path, fp, alpn, pbk, sid, spx…"""
    net = (p.get("type") or p.get("net") or "tcp").lower()
    if net in ("splithttp",):
        net = "xhttp"
    if net in ("h2", "http"):
        raise LinkError("انتقال h2/http در هستهٔ جدید Xray حذف شده؛ لینک xhttp یا ws بگیر.")
    if net == "quic":
        raise LinkError("انتقال quic در هستهٔ جدید Xray پشتیبانی نمی‌شود.")
    if net not in ("tcp", "raw", "ws", "grpc", "httpupgrade", "xhttp", "kcp", "mkcp"):
        raise LinkError(f"نوع انتقال «{net}» پشتیبانی نمی‌شود.")
    s: dict[str, Any] = {"network": {"raw": "tcp", "mkcp": "kcp"}.get(net, net)}
    path, h = unquote(p.get("path", "")), p.get("host", "")
    if s["network"] == "ws":
        s["wsSettings"] = {"path": path or "/", **({"host": h} if h else {})}
    elif s["network"] == "grpc":
        s["grpcSettings"] = {"serviceName": unquote(p.get("serviceName") or p.get("path") or ""),
                             "multiMode": p.get("mode") == "multi"}
    elif s["network"] == "httpupgrade":
        s["httpupgradeSettings"] = {"path": path or "/", **({"host": h} if h else {})}
    elif s["network"] == "xhttp":
        s["xhttpSettings"] = {"path": path or "/", **({"host": h} if h else {}), "mode": p.get("mode") or "auto"}
    elif s["network"] == "kcp":
        s["kcpSettings"] = {"header": {"type": p.get("headerType") or "none"}, **({"seed": p["seed"]} if p.get("seed") else {})}
    elif p.get("headerType") == "http":
        s["tcpSettings"] = {"header": {"type": "http", "request": {"path": [path or "/"], **({"headers": {"Host": [h]}} if h else {})}}}

    security = (p.get("security") or "").lower()
    sni = p.get("sni") or p.get("peer") or h or host
    if security == "tls":
        tls: dict[str, Any] = {"serverName": sni}
        if p.get("alpn"):
            tls["alpn"] = [a for a in unquote(p["alpn"]).split(",") if a]
        if p.get("fp"):
            tls["fingerprint"] = p["fp"]
        if p.get("allowInsecure") in ("1", "true"):
            tls["allowInsecure"] = True
        s["security"], s["tlsSettings"] = "tls", tls
    elif security == "reality":
        if not p.get("pbk"):
            raise LinkError("لینک reality کلید عمومی (pbk) ندارد.")
        s["security"] = "reality"
        s["realitySettings"] = {"serverName": sni, "fingerprint": p.get("fp") or "chrome", "publicKey": p["pbk"],
                                "shortId": p.get("sid", ""), "spiderX": unquote(p.get("spx", ""))}
    elif security not in ("", "none"):
        raise LinkError(f"امنیت «{security}» پشتیبانی نمی‌شود.")
    return s


def _vless(u) -> ParsedLink:
    p = _q(u.query)
    if not u.username or not u.hostname:
        raise LinkError("لینک vless ناقص است.")
    user = {"id": unquote(u.username), "encryption": p.get("encryption") or "none"}
    if p.get("flow"):
        user["flow"] = p["flow"]
    port = _port(u.port)
    out = {"protocol": "vless", "settings": {"vnext": [{"address": u.hostname, "port": port, "users": [user]}]},
           "streamSettings": _stream(p, u.hostname)}
    return ParsedLink("vless", unquote(u.fragment), u.hostname, port, out)


def _vmess(rest: str) -> ParsedLink:
    try:
        cfg = json.loads(_b64(rest.split("#", 1)[0]))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise LinkError("لینک vmess قابل خواندن نیست.") from None
    host, uid = str(cfg.get("add", "")).strip(), str(cfg.get("id", "")).strip()
    if not host or not uid:
        raise LinkError("لینک vmess ناقص است.")
    port = _port(cfg.get("port"))
    p = {"type": cfg.get("net") or "tcp", "headerType": cfg.get("type") or "", "host": cfg.get("host") or "",
         "path": cfg.get("path") or "", "security": cfg.get("tls") or "", "sni": cfg.get("sni") or "",
         "alpn": cfg.get("alpn") or "", "fp": cfg.get("fp") or "", "serviceName": cfg.get("path") or ""}
    if p["type"] == "kcp":
        p["seed"] = cfg.get("path") or ""
    user = {"id": uid, "alterId": int(cfg.get("aid") or 0), "security": cfg.get("scy") or "auto"}
    out = {"protocol": "vmess", "settings": {"vnext": [{"address": host, "port": port, "users": [user]}]},
           "streamSettings": _stream({k: str(v) for k, v in p.items()}, host)}
    return ParsedLink("vmess", str(cfg.get("ps") or ""), host, port, out)


def _trojan(u) -> ParsedLink:
    p = _q(u.query)
    if not u.username or not u.hostname:
        raise LinkError("لینک trojan ناقص است.")
    p.setdefault("security", "tls")
    port = _port(u.port)
    out = {"protocol": "trojan", "settings": {"servers": [{"address": u.hostname, "port": port, "password": unquote(u.username)}]},
           "streamSettings": _stream(p, u.hostname)}
    return ParsedLink("trojan", unquote(u.fragment), u.hostname, port, out)


def _ss(rest: str) -> ParsedLink:
    body, _, name = rest.partition("#")
    body = body.split("?", 1)[0].rstrip("/")
    try:
        if "@" in body:  # SIP002: base64(method:password)@host:port, or method:password in clear
            info, _, hostport = body.rpartition("@")
            info = unquote(info)
            userinfo = info if ":" in info else _b64(info)
        else:  # old form: base64(method:password@host:port)
            userinfo, _, hostport = _b64(body).rpartition("@")
    except (binascii.Error, UnicodeDecodeError):
        raise LinkError("لینک shadowsocks قابل خواندن نیست.") from None
    method, _, password = userinfo.partition(":")
    host, _, port_s = hostport.rpartition(":")
    host = host.strip("[]")
    if not (method and password and host):
        raise LinkError("لینک shadowsocks ناقص است.")
    port = _port(port_s)
    out = {"protocol": "shadowsocks", "settings": {"servers": [{"address": host, "port": port, "method": method, "password": password}]}}
    return ParsedLink("ss", unquote(name), host, port, out)


def _plain(link: str, u) -> ParsedLink:
    scheme = u.scheme.lower()
    if not u.hostname:
        raise LinkError("آدرس پروکسی ناقص است.")
    port = _port(u.port or (443 if scheme == "https" else 1080 if scheme.startswith("socks") else 80))
    auth = ""
    if u.username and u.password is None and scheme == "socks":
        try:  # v2rayN writes socks://base64(user:pass)@host:port
            user, _, pw = _b64(unquote(u.username)).partition(":")
            auth = f"{quote(user, safe='')}:{quote(pw, safe='')}@"
        except (binascii.Error, UnicodeDecodeError):
            auth = f"{u.username}@"
    elif u.username:
        auth = f"{u.username}:{u.password or ''}@"
    target = "socks5" if scheme.startswith("socks") else scheme
    host = f"[{u.hostname}]" if ":" in u.hostname else u.hostname
    return ParsedLink("socks" if target == "socks5" else "http", unquote(u.fragment), u.hostname, port,
                      proxy_url=f"{target}://{auth}{host}:{port}")


def parse_link(link: str) -> ParsedLink:
    link = link.strip()
    scheme, sep, rest = link.partition("://")
    scheme = scheme.lower()
    if not sep:
        raise LinkError("این خط لینک پروکسی نیست.")
    if scheme == "vmess":
        return _vmess(rest)
    if scheme == "ss":
        return _ss(rest)
    u = urlsplit(link)
    try:
        if scheme == "vless":
            return _vless(u)
        if scheme == "trojan":
            return _trojan(u)
        if scheme in ("socks", "socks5", "socks5h", "http", "https"):
            return _plain(link, u)
    except ValueError as exc:
        if isinstance(exc, LinkError):
            raise
        raise LinkError("لینک نامعتبر است.") from None
    if scheme in ("hysteria2", "hy2", "hysteria", "tuic", "wireguard", "wg"):
        raise LinkError(f"پروتکل {scheme} در Xray پشتیبانی نمی‌شود؛ vless، vmess، trojan یا shadowsocks بده.")
    raise LinkError(f"پروتکل «{scheme}» شناخته نشد.")


def split_links(text: str) -> list[str]:
    """Lines of a pasted list or a subscription body (often base64 of the whole list)."""
    text = text.strip()
    if text and "://" not in text:
        try:
            text = _b64("".join(text.split()))
        except (binascii.Error, UnicodeDecodeError):
            pass
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def xray_config(nodes: list[tuple[str, int, dict[str, Any]]]) -> dict[str, Any]:
    """One local SOCKS inbound per node (`tag`, local port, outbound), each routed to its own outbound."""
    inbounds, outbounds, rules = [], [], []
    for tag, port, outbound in nodes:
        inbounds.append({"tag": f"in-{tag}", "listen": "127.0.0.1", "port": port, "protocol": "socks",
                         "settings": {"udp": False}, "sniffing": {"enabled": False}})
        outbounds.append({**outbound, "tag": f"out-{tag}"})
        rules.append({"type": "field", "inboundTag": [f"in-{tag}"], "outboundTag": f"out-{tag}"})
    outbounds.append({"tag": "block", "protocol": "blackhole"})
    return {"log": {"loglevel": "warning"}, "inbounds": inbounds, "outbounds": outbounds,
            "routing": {"domainStrategy": "AsIs", "rules": rules}}
