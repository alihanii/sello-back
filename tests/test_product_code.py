from app.services.product_code import find_code, with_code_first
from tests.conftest import login


def test_find_code_in_text():
    assert find_code("مانتو لینن\nکد: A-120\nجنس لینن") == "A-120"
    assert find_code("کد محصول ۳۴۵ موجود است") == "345"
    assert find_code("SKU: hd_9") == "hd_9"
    assert find_code("#کد_۷۷") == "77"
    assert find_code("با کد تخفیف ارسال رایگان", "کدهای رنگی") == ""
    assert find_code("کد پستی را بفرستید") == ""
    assert find_code("", "یادداشت: کد ۱۲") == "12"


def test_code_goes_on_top_once():
    assert with_code_first("جنس لینن", "A1") == "کد محصول: A1\nجنس لینن"
    assert with_code_first("کد: A1\n\nجنس لینن\nکد: A1", "A1") == "کد محصول: A1\nجنس لینن"
    assert with_code_first("", "7") == "کد محصول: 7"
    assert with_code_first("کد محصول: 7\nمتن", "7") == "کد محصول: 7\nمتن"


async def test_new_products_get_a_code(client):
    h = await login(client, "09127770101")
    sid = (await client.post("/stores", json={"name": "کدها"}, headers=h)).json()["id"]
    # From the id when the text has no code.
    p = (await client.post(f"/stores/{sid}/products", headers=h, json={"title": "ماگ", "description": "سفید"})).json()
    assert p["sku"] == str(p["id"])
    assert p["description"] == f"کد محصول: {p['id']}\nسفید"
    # From the description when it has one.
    q = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "کیف", "description": "چرم طبیعی\nکد: K-55"})).json()
    assert q["sku"] == "K-55" and q["description"] == "کد محصول: K-55\nچرم طبیعی"
    # A code the seller typed wins, and is put on top too.
    r = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "کفش", "sku": "SH1", "description": "سبک"})).json()
    assert r["sku"] == "SH1" and r["description"].startswith("کد محصول: SH1\n")
    # A code already used by another product in the shop falls back to the id.
    s = (await client.post(f"/stores/{sid}/products", headers=h,
                           json={"title": "کیف ۲", "description": "کد: K-55"})).json()
    assert s["sku"] == str(s["id"])
