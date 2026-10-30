from app.services.text import build_caption, parse_caption, parse_number


def test_parse_caption_markers():
    r = parse_caption("مانتو لینن تابستانه\nجنس: لینن\nقیمت: ۱,۲۵۰,۰۰۰ تومان\nموجودی: ۷\n#مانتو")
    assert r["title"] == "مانتو لینن تابستانه"
    assert r["price"] == 1_250_000
    assert r["quantity"] == 7
    assert r["complete"] and "#مانتو" in r["hashtags"]


def test_parse_single_line_caption():
    r = parse_caption("کیف چرم دست‌دوز قیمت: 890000 موجودی: 3")
    assert r["title"] == "کیف چرم دست‌دوز" and r["price"] == 890000 and r["quantity"] == 3


def test_parse_number_units():
    assert parse_number("۲ میلیون") == 2_000_000
    assert parse_number("450 هزار تومان") == 450_000


def test_caption_reserves_price():
    cap = build_caption(title="عنوان", description="x" * 2000, price=150000, show_price=True, signature="@shop")
    assert len(cap) <= 1000
    assert "۱۵۰٬۰۰۰ تومان" in cap and cap.endswith("@shop")
