"""Sales-chat scenarios against a local backend with a real model. Usage: python run.py [scenario ...]
Env: API (default http://127.0.0.1:8000/api/v1), AVALAI_KEY, MODEL (default gemini-2.5-flash-lite)."""
import json, os, sys, time
import httpx

API = os.environ.get("API", "http://127.0.0.1:8000/api/v1")
c = httpx.Client(base_url=API, timeout=120)


def login(phone):
    code = c.post("/auth/otp/send", json={"phone": phone}).json()["dev_code"]
    return {"Authorization": "Bearer " + c.post("/auth/otp/verify", json={"phone": phone, "code": code}).json()["access_token"]}


def setup():
    st = json.load(open(".state.json")) if os.path.exists(".state.json") else {}
    if st:
        return st
    a = login("09120000000")
    r = c.post("/admin/ai/providers", headers=a, json={"name": "avalai", "kind": "avalai", "api_key": os.environ["AVALAI_KEY"],
                                                      "default_model": os.environ.get("MODEL", "gemini-2.5-flash-lite")})
    assert r.status_code in (200, 201), r.text
    h = login("09121234567")
    sid = c.post("/stores", headers=h, json={"name": "تست هانی"}).json()["id"]
    for p in [
        {"title": "هودی چاپ دلخواه (طرح شما)", "price": 2000000, "stock_qty": 50, "category": "هودی", "sku": "PRINT-HOODIE",
         "tags": ["چاپ", "طرح دلخواه"],
         "description": "هودی با طرح دلخواه شما؛ طرح رو توی چت بفرستید تا روی جلو یا پشت هودی چاپ بشه. رنگ‌ها: مشکی، سفید، طوسی. سایزها: S، M، L، XL، 2XL."},
        {"title": "ست هودی اسپایدرمن", "price": 1800000, "stock_qty": 4, "category": "هودی", "attributes": {"سایز": "M، L، XL", "رنگ": "قرمز، مشکی"}},
        {"title": "هودی ساده مشکی", "price": 950000, "stock_qty": 10, "category": "هودی", "attributes": {"سایز": "S، M، L، XL"}},
        {"title": "تیشرت نخی سفید", "price": 450000, "stock_qty": 0, "category": "تیشرت", "attributes": {"سایز": "M، L"}},
        {"title": "ماگ سرامیکی", "price": 250000, "stock_qty": 20, "category": "ماگ"},
    ]:
        assert c.post(f"/stores/{sid}/products", headers=h, json=p).status_code in (200, 201)
    r = c.put(f"/stores/{sid}/chat/agent", headers=h, json={
        "enabled": True, "orders_enabled": True,
        "payment": {"card_number": "6037-9911-2233-4455", "holder": "علی هانی", "require_receipt": True},
        "faqs": [{"q": "هزینه ارسال چقدره؟", "a": "ارسال با پست پیشتاز ۸۰ هزار تومان؛ بالای ۳ میلیون رایگان."}]})
    assert r.status_code == 200, r.text
    st = {"h": h, "sid": sid}
    json.dump(st, open(".state.json", "w"))
    return st


SCENARIOS = {
    "print": ["سلام", "هودی با طرح دلخواه چاپ میکنی؟", "@photo:spider.png|این", "سایز بندی چی داری؟", "مشکی XL پشت",
              "نه همین", "علی رضایی ۰۹۱۲۳۴۵۶۷۸۹ تهران خیابان ولیعصر پلاک ۱۲ کدپستی ۱۲۳۴۵۶۷۸۹۰", "بله"],
    "change": ["سلام هودی ساده مشکی دارین؟", "یه L میخوام", "نه ببخشید XL باشه", "دوتا کن",
               "یه ماگ هم اضافه کن", "ماگ رو حذف کن", "سارا محمدی 09351112233 اصفهان خیابان چهارباغ پلاک ۵ کدپستی ۸۱۲۳۴۵۶۷۸۹",
               "شماره‌ام اشتباه شد 09351112234 درستشه", "بله ثبت کن"],
    "hard": ["تیشرت سفید M دارین؟", "کی موجود میشه؟", "باشه اسپایدرمن قرمز سایز S بده", "پس M",
             "ارزونتر نمیدی؟", "هزینه ارسال چقدره", "یه شعر برام بگو", "کنسلش کن"],
    "phone": ["هودی ساده مشکی M", "ثبت کن", "محمد", "0912", "۰۹۱۲۳۴۵۶۷۸۹", "تهران پاسداران کوچه ۳ پلاک ۷ کدپستی ۱۹۸۷۶۵۴۳۲۱", "بله"],
}


def run(name, st):
    h, sid = st["h"], st["sid"]
    c.get(f"/stores/{sid}/chat-link", headers=h)
    slug, visitor, tokens = f"shop{sid}", "", 0
    print(f"\n===== {name} =====")
    for msg in SCENARIOS[name]:
        if msg.startswith("@photo:"):
            path, _, text = msg[7:].partition("|")
            r = c.post(f"/public/chat/{slug}/photo", data={"name": "علی", "visitor": visitor, "text": text},
                       files={"file": (path, open(path, "rb"), "image/png")})
            msg = f"[عکس {path}] {text}"
        else:
            r = c.post(f"/public/chat/{slug}/messages", json={"name": "علی", "visitor": visitor, "text": msg})
        if r.status_code != 200:
            print("  !!", r.status_code, r.text[:200]); continue
        visitor = r.json()["visitor"]
        print(f"👤 {msg}")
        for m in r.json()["messages"][1:]:
            b = f"   [{' | '.join(sum(m['buttons'] or [], []))}]" if m.get("buttons") else ""
            print(f"🤖 {m['text']}{b}" + (" 🖼" if m.get("image_url") else ""))
    convs = c.get(f"/stores/{sid}/chat/conversations?platform=link", headers=h).json()
    rows = convs["items"] if isinstance(convs, dict) else convs
    conv = next((x for x in rows if x.get("external_chat_id") == visitor), None) or max(rows, key=lambda x: x["id"])
    d = c.get(f"/chat/conversations/{conv['id']}", headers=h).json()
    tokens = sum(m.get("input_tokens", 0) + m.get("output_tokens", 0) for m in d["messages"])
    print(f"-- state.order: {json.dumps((d['conversation'].get('state') or {}).get('order'), ensure_ascii=False)[:500]}")
    print(f"-- tokens: {tokens}")


if __name__ == "__main__":
    st = setup()
    for n in sys.argv[1:] or list(SCENARIOS):
        run(n, st)
