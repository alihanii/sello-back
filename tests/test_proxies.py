import base64
import json

import httpx
import pytest

from app.services import proxy_pool
from app.services.proxy_links import LinkError, parse_link, split_links, xray_config
from tests.conftest import login

VLESS_REALITY = ("vless://1b2c3d4e-0000-4000-8000-111122223333@de.example.com:443?type=tcp&security=reality"
                 "&sni=www.speedtest.net&fp=chrome&pbk=PUBKEY123&sid=ab12&flow=xtls-rprx-vision#Germany%201")
VLESS_WS = ("vless://uuid-1@cdn.example.com:8443?type=ws&security=tls&host=ws.example.com&path=%2Fray%3Fed%3D2048"
            "&sni=ws.example.com&alpn=h2%2Chttp%2F1.1#WS")


def test_vless_reality_and_ws():
    p = parse_link(VLESS_REALITY)
    assert (p.protocol, p.name, p.address, p.needs_xray) == ("vless", "Germany 1", "de.example.com:443", True)
    user = p.outbound["settings"]["vnext"][0]["users"][0]
    assert user == {"id": "1b2c3d4e-0000-4000-8000-111122223333", "encryption": "none", "flow": "xtls-rprx-vision"}
    s = p.outbound["streamSettings"]
    assert s["security"] == "reality" and s["realitySettings"]["publicKey"] == "PUBKEY123"
    assert s["realitySettings"]["serverName"] == "www.speedtest.net" and s["realitySettings"]["shortId"] == "ab12"

    s = parse_link(VLESS_WS).outbound["streamSettings"]
    assert s["network"] == "ws" and s["wsSettings"] == {"path": "/ray?ed=2048", "host": "ws.example.com"}
    assert s["tlsSettings"] == {"serverName": "ws.example.com", "alpn": ["h2", "http/1.1"]}


def test_vmess_trojan_and_shadowsocks():
    vm = {"v": "2", "ps": "IR→NL", "add": "nl.example.com", "port": "2053", "id": "uuid-2", "aid": "0", "scy": "auto",
          "net": "grpc", "type": "none", "host": "", "path": "grpcsvc", "tls": "tls", "sni": "nl.example.com"}
    p = parse_link("vmess://" + base64.b64encode(json.dumps(vm).encode()).decode())
    assert (p.protocol, p.name, p.port) == ("vmess", "IR→NL", 2053)
    assert p.outbound["streamSettings"]["grpcSettings"] == {"serviceName": "grpcsvc", "multiMode": False}

    t = parse_link("trojan://pa%40ss@tr.example.com:443?sni=tr.example.com#TR")
    assert t.outbound["settings"]["servers"][0]["password"] == "pa@ss"
    assert t.outbound["streamSettings"]["security"] == "tls"

    userinfo = base64.urlsafe_b64encode(b"chacha20-ietf-poly1305:secret").decode().rstrip("=")
    ss = parse_link(f"ss://{userinfo}@1.2.3.4:8388#SS")
    assert ss.outbound["settings"]["servers"][0] == {"address": "1.2.3.4", "port": 8388,
                                                      "method": "chacha20-ietf-poly1305", "password": "secret"}
    old = parse_link("ss://" + base64.b64encode(b"aes-256-gcm:pw@5.6.7.8:443").decode() + "#old")
    assert old.address == "5.6.7.8:443" and old.outbound["settings"]["servers"][0]["method"] == "aes-256-gcm"


def test_plain_proxies_need_no_xray():
    s = parse_link("socks://" + base64.b64encode(b"user:pass").decode() + "@9.9.9.9:1080#s")
    assert (s.protocol, s.needs_xray, s.proxy_url) == ("socks", False, "socks5://user:pass@9.9.9.9:1080")
    h = parse_link("http://u:p@proxy.example.com:3128")
    assert h.proxy_url == "http://u:p@proxy.example.com:3128"


@pytest.mark.parametrize("link", ["hysteria2://x@h:443", "vless://u@h:443?type=h2", "nonsense", "vless://u@h:99999",
                                  "vless://u@h:443?security=reality"])
def test_bad_links_are_explained(link):
    with pytest.raises(LinkError):
        parse_link(link)


def test_subscription_body_and_xray_config():
    body = base64.b64encode(f"{VLESS_REALITY}\n# comment\n{VLESS_WS}\n".encode()).decode()
    assert split_links(body) == [VLESS_REALITY, VLESS_WS]
    cfg = xray_config([("7", 20001, parse_link(VLESS_WS).outbound)])
    assert cfg["inbounds"][0]["port"] == 20001 and cfg["inbounds"][0]["listen"] == "127.0.0.1"
    assert cfg["routing"]["rules"][0] == {"type": "field", "inboundTag": ["in-7"], "outboundTag": "out-7"}


def _node(nid, purposes, health=None, url="socks5://127.0.0.1:1"):
    return proxy_pool._Node(nid, f"n{nid}", "socks", purposes, proxy_url=url, health=health or {})  # noqa: SLF001


def test_routes_prefer_fast_healthy_nodes_and_respect_modes(monkeypatch):
    p = proxy_pool.Pool()
    p.nodes = {1: _node(1, ["telegram"], {"telegram": {"ok": True, "latency_ms": 900}}, "socks5://a:1"),
               2: _node(2, ["telegram"], {"telegram": {"ok": True, "latency_ms": 120}}, "socks5://b:1"),
               3: _node(3, ["telegram"], {"telegram": {"ok": False}}, "socks5://c:1"),
               4: _node(4, ["instagram"], {}, "socks5://d:1")}
    assert p.routes("telegram") == [(2, "socks5://b:1"), (1, "socks5://a:1"), (None, None)]
    assert p.routes("instagram", "http://legacy:1") == [(4, "socks5://d:1"), (None, "http://legacy:1"), (None, None)]
    p.modes["telegram"] = "proxy"
    assert p.routes("telegram")[-1] == (1, "socks5://a:1")  # never direct
    p.modes["telegram"] = "direct"
    assert p.routes("telegram") == [(None, None)]
    p.modes["telegram"] = "auto"
    p.report(2, "telegram", False, error="boom")
    assert p.routes("telegram")[0] == (1, "socks5://a:1")


async def test_request_fails_over_to_the_next_node(monkeypatch):
    p = proxy_pool.Pool()
    p.nodes = {1: _node(1, ["telegram"], {"telegram": {"ok": True, "latency_ms": 50}}, "socks5://dead:1"),
               2: _node(2, ["telegram"], {"telegram": {"ok": True, "latency_ms": 80}}, "socks5://alive:1")}
    monkeypatch.setattr(proxy_pool, "pool", p)
    tried = []

    class FakeClient:
        def __init__(self, proxy=None, **kw):
            self.proxy = proxy

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kw):
            tried.append(self.proxy)
            if self.proxy == "socks5://dead:1":
                raise httpx.ConnectError("tls handshake failed")
            return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(proxy_pool.httpx, "AsyncClient", FakeClient)
    resp = await proxy_pool.request("telegram", "POST", "https://api.telegram.org/botX/getMe", json={})
    assert resp.json() == {"ok": True} and tried == ["socks5://dead:1", "socks5://alive:1"]
    assert p.nodes[1].health["telegram"]["ok"] is False and p.recheck.is_set()
    assert p.routes("telegram")[0][0] == 2


async def test_admin_import_modes_and_delete(client):
    h = await login(client, "09120000000")
    text = "\n".join(["socks5://1.1.1.1:1080#first", "http://2.2.2.2:3128", "socks5://1.1.1.1:1080#first",
                      "hysteria2://x@h:443", VLESS_REALITY])
    r = await client.post("/admin/proxies/import", headers=h, json={"text": text, "purposes": ["telegram"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added"] == 3 and len(body["skipped"]) == 2
    nodes = {n["address"]: n for n in body["nodes"]}
    assert nodes["1.1.1.1:1080"]["name"] == "first" and nodes["1.1.1.1:1080"]["purposes"] == ["telegram"]
    assert "link" not in nodes["1.1.1.1:1080"]
    if not body["xray"]["installed"]:
        assert "Xray" in nodes["de.example.com:443"]["error"]

    r = await client.put("/admin/proxies/modes", headers=h, json={"modes": {"instagram": "direct"}})
    assert r.status_code == 200
    assert {p["id"]: p["mode"] for p in r.json()["purposes"]} == {"telegram": "auto", "instagram": "direct", "linkedin": "auto"}
    assert proxy_pool.pool.routes("instagram") == [(None, None)]

    nid = nodes["2.2.2.2:3128"]["id"]
    r = await client.patch(f"/admin/proxies/{nid}", headers=h, json={"purposes": ["telegram", "instagram"], "enabled": False})
    assert r.status_code == 200 and nid not in proxy_pool.pool.nodes
    r = await client.delete(f"/admin/proxies/{nid}", headers=h)
    assert r.status_code == 200 and all(n["id"] != nid for n in r.json()["nodes"])

    seller = await login(client, "09125559999")
    assert (await client.get("/admin/proxies", headers=seller)).status_code == 403
    await client.put("/admin/proxies/modes", headers=h, json={"modes": {"instagram": "auto"}})
    for n in (await client.get("/admin/proxies", headers=h)).json()["nodes"]:
        await client.delete(f"/admin/proxies/{n['id']}", headers=h)
