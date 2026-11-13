"""Temporary: read Digikala's seller PWA from the server (no login, no SMS) to find the current sign-in calls."""
import asyncio
import re

import httpx

B = "https://seller.digikala.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0 Safari/537.36"}


async def main():
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=UA) as c:
        r = await c.get(f"{B}/pwa/account/sign-in")
        print("SIGNIN", r.status_code, r.url, r.headers.get("content-type"), "cookies:", list(r.cookies.keys()))
        html = r.text
        print("HTML_HEAD", html[:1200].replace("\n", " "))
        srcs = sorted(set(re.findall(r'src="([^"]+\.js)"', html)))
        print("SCRIPTS", srcs)
        texts = {}
        for s in srcs:
            u = s if s.startswith("http") else B + s
            texts[u] = (await c.get(u)).text
        chunks = set()
        for t in list(texts.values()):
            for m in re.finditer(r'static/js/async/"\+\(?(\{[^}]*\})?[^+]*\+"\."\+(\{[^}]*\})', t):
                names = dict(re.findall(r'"?([\w-]+)"?:"([\w./-]+)"', m.group(1) or ""))
                hashes = dict(re.findall(r'"?([\w-]+)"?:"([0-9a-f]+)"', m.group(2)))
                for k, h in hashes.items():
                    chunks.add(f"{B}/pwa/static/js/async/{names.get(k, k)}.{h}.js")
        print("CHUNKS", len(chunks))
        sem = asyncio.Semaphore(12)

        async def get(u):
            async with sem:
                try:
                    texts[u] = (await c.get(u)).text
                except Exception as e:  # noqa: BLE001
                    print("FAIL", u, e)

        await asyncio.gather(*(get(u) for u in chunks))
        print("DOWNLOADED", len(texts))
        seen = set()
        for u, t in texts.items():
            for w in ("otp/", "/otp", "login", "captcha", "challenge_token", "verify"):
                for m in re.finditer(re.escape(w), t):
                    snip = t[max(0, m.start() - 350): m.end() + 350]
                    key = snip[300:420]
                    if key in seen or not re.search(r"post|url|captcha|otp", snip, re.I):
                        continue
                    seen.add(key)
                    print("SNIP", u.rsplit("/", 1)[-1], w, "::", snip.replace("\n", " "))
                    if len(seen) > 150:
                        break
        hdr = {"Content-Type": "application/json", "Accept": "application/json", "Origin": B,
               "Referer": f"{B}/pwa/account/sign-in", "x-web-optimize-response": "1"}
        r = await c.post(f"{B}/api/v2/login", json={"username": "abc", "captcha_token": ""}, headers=hdr)
        print("PROBE login", r.status_code, dict(r.headers), r.text[:1500])
        r = await c.get(f"{B}/api/v2/profile?mini=true", headers=hdr)
        print("PROBE profile", r.status_code, r.text[:500])


asyncio.run(main())
