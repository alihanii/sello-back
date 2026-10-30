"""Fill a dev database with a realistic demo shop (channels, products, logs, 60 days of
activity) so every page has something to show. Never run against production.

    python -m scripts.demo_seed          # from backend/
Demo logins: 09121234567 (seller), 09120000000 (admin); the dev OTP is shown on the login page.
"""

import asyncio
import random
import struct
import zlib
from datetime import timedelta

from app.core.db import SessionLocal, init_db, utcnow
from app.core.security import encrypt_credentials
from app.models import (
    ActivityEvent, AiProvider, AiTaskRoute, AiUsageLog, Approval, Automation, Channel, ChannelListing, CrawlItem,
    CrawlSource, DEFAULT_STORE_SETTINGS, Operation, OperationStep, Plan, PlanRequest, Product, ProductImage, Store,
    StoreStyleProfile, Subscription, User,
)
from app.seed import seed
from app.services.media import save_bytes

random.seed(7)

PRODUCTS = [
    ("مانتو لینن تابستانه مدل رها", 1_250_000, 7, "#C8B6A6"), ("شال نخی ساده ۵ رنگ", 350_000, 24, "#9DB4C0"),
    ("کیف دوشی چرم دست‌دوز", 2_890_000, 3, "#8B5E3C"), ("شومیز حریر گلدار", 780_000, 0, "#E8A0BF"),
    ("شلوار جین مام استایل", 1_490_000, 11, "#5B7DB1"), ("روسری ابریشم طرح مینیاتور", 950_000, 5, "#D4A373"),
    ("کفش اسپرت زنانه سبک", 1_980_000, 2, "#EDEDED"), ("هودی اورسایز پنبه", 1_150_000, 9, "#6B705C"),
    ("دامن پلیسه میدی", 890_000, 6, "#A3B18A"), ("ست گردنبند و دستبند نقره", 1_680_000, 4, "#B8B8D1"),
    ("کت تک کتان", 2_350_000, 1, "#3D405B"), ("تیشرت یقه گرد بیسیک", 420_000, 30, "#F2CC8F"),
]
DESC = ("جنس: {m}\nسایزبندی: ۳۸ تا ۴۶\nرنگ‌بندی متنوع، مناسب استفادهٔ روزمره و مهمانی.\n"
        "دوخت تمیز و تمیزکاری کامل؛ قبل از ارسال کنترل کیفیت می‌شود.\nارسال به سراسر ایران، تحویل ۲ تا ۴ روز کاری.")


def png(color: str, size: int = 360) -> bytes:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    rows = []
    for y in range(size):
        k = 0.75 + 0.25 * y / size
        row = bytes([0]) + bytes([min(255, int(r * k)), min(255, int(g * k)), min(255, int(b * k))]) * size
        rows.append(row)
    raw = zlib.compress(b"".join(rows), 9)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def step(op, order, name, label, req, resp, ms=400, status="succeeded", msg=""):
    return OperationStep(operation_id=op.id, order=order, name=name, label_fa=label, status=status, started_at=op.started_at,
                         duration_ms=ms + random.randint(0, 300), request=req, response=resp, message_fa=msg)


async def main() -> None:
    await init_db()
    await seed()
    now = utcnow()
    async with SessionLocal() as db:
        admin = User(phone="09120000000", name="ادمین", role="admin", onboarded=True, is_active=True, last_login_at=now)
        seller = User(phone="09121234567", name="علی", role="seller", onboarded=True, is_active=True, last_login_at=now)
        others = [User(phone=f"0912{random.randint(1000000, 9999999)}", name=n, role="seller", onboarded=True, is_active=True,
                       last_login_at=now - timedelta(days=random.randint(0, 20))) for n in ("سارا", "رضا", "نگار", "حمید")]
        db.add_all([admin, seller, *others])
        await db.flush()
        pro = next(p for p in (await db.execute(Plan.__table__.select())).all() if p.tier == "pro" and p.duration_months == 3)
        db.add(Subscription(user_id=seller.id, plan_id=pro.id, starts_at=now - timedelta(days=12), ends_at=now + timedelta(days=78), activated_by=f"admin:{admin.id}"))
        db.add(PlanRequest(user_id=others[0].id, plan_id=pro.id))

        store = Store(owner_id=seller.id, name="بوتیک مهتاب", category="پوشاک زنانه",
                      settings={**DEFAULT_STORE_SETTINGS, "caption_signature": "🛍 سفارش در دایرکت | ارسال به سراسر ایران"})
        store2 = Store(owner_id=seller.id, name="اکسسوری ماه", category="اکسسوری", settings=dict(DEFAULT_STORE_SETTINGS))
        other_stores = [Store(owner_id=u.id, name=n, category="", settings=dict(DEFAULT_STORE_SETTINGS))
                        for u, n in zip(others, ("کفش سارا", "گجت رضا", "گالری نگار", "عطر حمید"))]
        db.add_all([store, store2, *other_stores])
        await db.flush()

        creds = encrypt_credentials({"bot_token": "demo"})
        tg = Channel(store_id=store.id, platform="telegram", display_name="بوتیک مهتاب | تلگرام", external_account_id="-1001234",
                     credentials_encrypted=creds, config={"chat_id": -1001234, "username": "mahtab_boutique"}, status="connected", last_health_at=now)
        bale = Channel(store_id=store.id, platform="bale", display_name="بوتیک مهتاب | بله", external_account_id="5512",
                       credentials_encrypted=creds, config={"chat_id": 5512, "username": "mahtab"}, status="connected", last_health_at=now - timedelta(hours=3))
        dk = Channel(store_id=store.id, platform="digikala", display_name="مهتاب در دیجی‌کالا", external_account_id="88231",
                     credentials_encrypted=encrypt_credentials({"access_token": "demo"}), config={"seller_id": 88231},
                     status="expired", last_error="نشست دیجی‌کالا منقضی شده؛ دوباره با کد تأیید وارد شوید.", last_health_at=now - timedelta(days=1))
        db.add_all([tg, bale, dk])
        for s in other_stores[:3]:
            db.add(Channel(store_id=s.id, platform=random.choice(["telegram", "bale"]), display_name=f"{s.name} | کانال", credentials_encrypted=creds,
                           config={"chat_id": 1}, status=random.choice(["connected", "connected", "error"]), last_error=""))
        await db.flush()

        src = CrawlSource(store_id=store.id, name="پست‌های کانال تلگرام", type="channel_posts", channel_id=tg.id, schedule="daily",
                          extraction_mode="auto", last_run_at=now - timedelta(hours=5), next_run_at=now + timedelta(hours=19),
                          created_at=now - timedelta(days=45))
        db.add(src)
        await db.flush()

        products = []
        for i, (title, price, qty, color) in enumerate(PRODUCTS):
            source = ["manual", "crawl", "crawl", "ai"][i % 4]
            p = Product(store_id=store.id, title=title, description=DESC.format(m=random.choice(["لینن", "نخ پنبه", "حریر", "چرم طبیعی"])),
                        price=price, compare_at_price=int(price * 1.2) if i % 3 == 0 else 0, stock_qty=qty, sku=f"MH-{1000 + i}",
                        source=source, created_by={"manual": "user", "crawl": "crawler", "ai": "ai"}[source],
                        tags=random.sample(["تابستانه", "جدید", "حراج", "پرفروش"], 2), status="ready", is_golden_sample=i in (0, 2),
                        attributes={"جنس": "لینن", "سایز": "۳۸–۴۶"} if i % 2 == 0 else {},
                        source_ref={"key": f"src:{src.id}", "external_id": str(700 + i), "url": f"https://t.me/mahtab_boutique/{700 + i}"} if source == "crawl"
                        else {"field_sources": {"title": "image", "description": "style", "attributes": "image", "price": "note"}} if source == "ai" else {},
                        created_at=now - timedelta(days=40 - i * 3))
            db.add(p)
            await db.flush()
            for k in range(1 + i % 3):
                db.add(ProductImage(product_id=p.id, url=save_bytes(store.id, png(color if k == 0 else "#DDDDDD")), position=k))
            products.append(p)
        draft = Product(store_id=store.id, title="بلوز کرپ آستین پفی", description="کرپ مازراتی، سایز فری", price=690_000, stock_qty=8,
                        source="crawl", created_by="crawler", status="draft", source_ref={"key": f"src:{src.id}", "external_id": "791"})
        db.add(draft)
        await db.flush()
        db.add(ProductImage(product_id=draft.id, url=save_bytes(store.id, png("#F4A261")), position=0))

        # listings + publish operations with step logs
        for i, p in enumerate(products):
            for ch in (tg, bale):
                if (i + ch.id) % 3 == 0:
                    continue
                failed = (i == 4 and ch is bale)
                by = "ai" if i % 2 else "user"
                l = ChannelListing(product_id=p.id, channel_id=ch.id, status="failed" if failed else ("out_of_sync" if i == 1 and ch is tg else "published"),
                                   remote_id="" if failed else str(800 + i), remote_url="" if failed else f"https://{'t.me' if ch is tg else 'ble.ir'}/{ch.config['username']}/{800 + i}",
                                   published_by="" if failed else by, published_at=None if failed else now - timedelta(days=30 - i * 2, hours=i),
                                   last_error_fa="ربات در کانال ادمین نیست یا اجازهٔ ارسال پست ندارد." if failed else "")
                db.add(l)
                await db.flush()
                t = now - timedelta(days=30 - i * 2, hours=i)
                op = Operation(store_id=store.id, type="publish", title_fa=f"انتشار «{p.title}» در {ch.display_name}", channel_id=ch.id, product_id=p.id,
                               ref_type="listing", ref_id=l.id, trigger="automation" if by == "ai" else "user", actor=by,
                               status="failed" if failed else "succeeded", progress=100, created_at=t, started_at=t, finished_at=t + timedelta(seconds=3), seen=not failed,
                               error_code="bot_not_admin" if failed else "", error_message_fa=l.last_error_fa,
                               suggested_action={"type": "help", "label": "راهنمای ادمین کردن ربات"} if failed else {},
                               result={} if failed else {"remote_id": l.remote_id, "remote_url": l.remote_url})
                db.add(op)
                await db.flush()
                db.add(step(op, 1, "validate", "بررسی محصول با قوانین پلتفرم", {"title": p.title, "images": 1 + i % 3}, {"errors": []}, 3))
                db.add(step(op, 2, "build_caption", "آماده‌سازی متن پست", {}, {"caption": f"{p.title}\n\n…\n\n💰 قیمت: {price:,} تومان", "length": 312}, 2))
                db.add(step(op, 3, "load_images", f"آماده‌سازی {1 + i % 3} عکس", {"images": [im.url for im in []] or ["/media/..."]}, {"loaded": 1 + i % 3}, 120))
                if failed:
                    db.add(step(op, 4, "send", "ارسال آلبوم به کانال", {"method": "sendMediaGroup", "chat_id": ch.config["chat_id"]},
                                {"ok": False, "error_code": 403, "description": "Forbidden: bot is not a member of the channel chat"}, 600, "failed", l.last_error_fa))
                else:
                    db.add(step(op, 4, "send", "ارسال آلبوم به کانال", {"method": "sendMediaGroup", "chat_id": ch.config["chat_id"]},
                                [{"message_id": 800 + i, "chat": {"id": ch.config["chat_id"]}, "date": int(t.timestamp())}], 900))

        # crawl runs
        crawl_op = Operation(store_id=store.id, type="crawl", title_fa="کرال «پست‌های کانال تلگرام»", channel_id=tg.id, ref_type="crawl_source", ref_id=src.id,
                             trigger="schedule", actor="ai", status="partial", progress=100, created_at=now - timedelta(hours=5), started_at=now - timedelta(hours=5),
                             finished_at=now - timedelta(hours=5) + timedelta(seconds=14), error_message_fa="۱ آیتم از ۶ آیتم استخراج نشد.",
                             result={"seen": 6, "created": 2, "updated": 1, "duplicate": 1, "skipped": 1, "failed": 1, "inbox": 0})
        db.add(crawl_op)
        await db.flush()
        db.add(step(crawl_op, 1, "get_updates", "دریافت پست‌های جدید کانال", {"method": "getUpdates", "offset": 9120}, {"count": 6}, 700))
        db.add(step(crawl_op, 2, "extract", "استخراج دیتای ۶ آیتم", {"mode": "auto"}, {"methods": {"701": "regex", "702": "llm", "703": "regex"}}, 4200))
        db.add(step(crawl_op, 3, "save", "ساخت/به‌روزرسانی محصولات", {}, crawl_op.result, 300))
        for ext, res, title, err in [("701", "created", PRODUCTS[1][0], ""), ("702", "created", PRODUCTS[5][0], ""), ("703", "updated", PRODUCTS[2][0], ""),
                                     ("704", "duplicate", PRODUCTS[9][0], ""), ("705", "skipped", "", "این پست شبیه محصول نبود (نه قیمت دارد نه عکس)."),
                                     ("706", "failed", "", "عنوان محصول از متن پیدا نشد.")]:
            db.add(CrawlItem(operation_id=crawl_op.id, source_id=src.id, external_id=ext, result=res, error_fa=err,
                             raw={"caption": f"{title or 'سلام دوستان، تخفیف‌های آخر هفته در راه است'}\nقیمت: ۹۵۰,۰۰۰\nموجودی: ۵" if title else "سلام دوستان 🌸"},
                             extracted={"title": title, "method": "llm" if ext == "702" else "regex"}))

        dk_op = Operation(store_id=store.id, type="health_check", title_fa="بررسی سلامت مهتاب در دیجی‌کالا", channel_id=dk.id, status="failed",
                          created_at=now - timedelta(days=1), started_at=now - timedelta(days=1), finished_at=now - timedelta(days=1), error_code="expired",
                          error_message_fa=dk.last_error, suggested_action={"type": "reconnect", "label": "اتصال مجدد"}, trigger="schedule", actor="ai")
        db.add(dk_op)
        await db.flush()
        db.add(step(dk_op, 1, "profile", "بررسی نشست دیجی‌کالا", {}, {"status": 401}, 500, "failed", dk.last_error))

        running = Operation(store_id=store.id, type="reference", title_fa="ساخت محصول با AI از رفرنس", actor="ai", status="running",
                            current_step="ساخت محصول با هوش مصنوعی", progress=60, created_at=now - timedelta(seconds=20), started_at=now - timedelta(seconds=20))
        db.add(running)
        await db.flush()
        db.add(step(running, 1, "link_1", "کرال لینک ۱ از ۱", {"url": "https://example-shop.ir/p/1203"}, {"method": "json-ld", "title": "Linen Dress"}, 1800))
        db.add(step(running, 2, "style", "خواندن سبک فروشگاه", {}, {"has_profile": True, "samples": 5}, 30))

        # 60 days of calendar activity
        for d in range(60, -1, -1):
            day = (now - timedelta(days=d)).date()
            if d % 11 == 4:
                continue  # a scheduled crawl that didn't run
            items = max(0, int(random.gauss(14, 9))) if d % 9 != 2 else 0
            db.add(ActivityEvent(store_id=store.id, day=day, type="crawl_run", count=1))
            db.add(ActivityEvent(store_id=store.id, day=day, type="crawl_item", count=items))
            db.add(ActivityEvent(store_id=store.id, day=day, type="product_created", count=items // 3))
            db.add(ActivityEvent(store_id=store.id, day=day, type="publish_ok", count=random.randint(0, 9)))
            if random.random() < 0.18:
                db.add(ActivityEvent(store_id=store.id, day=day, type="publish_failed", count=random.randint(1, 2)))

        db.add(StoreStyleProfile(store_id=store.id, profile={
            "title_pattern": "[نوع محصول] [جنس] [ویژگی] مدل [نام]", "tone": "friendly", "description_length": "medium",
            "caption_structure": ["title", "features", "sizes", "price", "shipping", "cta"], "common_hashtags": ["#مانتو", "#پوشاک_زنانه", "#بوتیک_مهتاب"],
            "common_emojis": ["🌸", "✨", "🛍"], "always_mentions": ["سایزبندی", "ارسال به سراسر ایران"], "price_ranges": {"مانتو": [900000, 2500000]},
            "example_title": "مانتو لینن تابستانه مدل رها"}, sample_product_ids=[p.id for p in products[:5]]))
        rule = Automation(store_id=store.id, name="انتشار خودکار محصولات کرال‌شده", trigger="product_created",
                          conditions={"sources": ["crawl"], "has_price": True, "min_images": 1}, actions={"channel_ids": [tg.id, bale.id], "mode": "auto"},
                          run_count=37, last_run_at=now - timedelta(hours=5))
        rule2 = Automation(store_id=store.id, name="هر روز ساعت ۱۰ یک محصول در بله", trigger="schedule", conditions={"in_stock": True},
                           actions={"channel_ids": [bale.id], "mode": "approval", "schedule_hour": 10}, run_count=12, last_run_at=now - timedelta(hours=8))
        db.add_all([rule, rule2])
        await db.flush()
        db.add(Approval(store_id=store.id, product_id=products[6].id, automation_id=rule2.id, channel_ids=[bale.id]))

        # AI keys + 30 days of usage
        metis = AiProvider(name="Gemini اصلی (دمو)", kind="gemini", base_url="https://generativelanguage.googleapis.com", api_key_encrypted=encrypt_credentials({"api_key": "demo"}),
                           key_hint="8f2a", default_model="gemini-2.5-flash", input_price_per_m=900_000, output_price_per_m=3_600_000,
                           monthly_budget_toman=5_000_000, last_test_at=now - timedelta(hours=2), last_test_ok=True, last_test_message="کلید سالم است و پاسخ داد.")
        backup = AiProvider(name="واسط OpenAI (پشتیبان)", kind="openai", base_url="https://api.openai.com/v1", api_key_encrypted=encrypt_credentials({"api_key": "demo"}),
                            key_hint="x91Q", default_model="gpt-4.1-mini", input_price_per_m=1_200_000, output_price_per_m=4_800_000, monthly_budget_toman=2_000_000)
        db.add_all([metis, backup])
        await db.flush()
        for task in ("caption_extract", "product_generate", "style_profile"):
            db.add(AiTaskRoute(task=task, provider_id=metis.id, model="gemini-2.5-flash", fallback_provider_id=backup.id, fallback_model="gpt-4.1-mini"))
        for d in range(30):
            for _ in range(random.randint(8, 40)):
                prov = metis if random.random() < 0.85 else backup
                ok = random.random() > 0.05
                tin, tout = random.randint(400, 3000), random.randint(150, 900)
                db.add(AiUsageLog(created_at=now - timedelta(days=d, minutes=random.randint(0, 1400)), provider_id=prov.id,
                                  task=random.choice(["caption_extract", "caption_extract", "product_generate", "style_profile"]),
                                  model=prov.default_model, store_id=random.choice([store.id, store.id, other_stores[0].id]), ok=ok,
                                  is_fallback=prov is backup, latency_ms=random.randint(900, 6000), input_tokens=tin if ok else 0,
                                  output_tokens=tout if ok else 0, cost_toman=(tin * prov.input_price_per_m + tout * prov.output_price_per_m) // 1_000_000 if ok else 0,
                                  error_code="" if ok else random.choice(["ai_rate_limited", "ai_http_503"]),
                                  error_message="" if ok else "سقف استفاده از کلید API پر شده است."))
        await db.commit()
    print("demo data ready")


if __name__ == "__main__":
    asyncio.run(main())
