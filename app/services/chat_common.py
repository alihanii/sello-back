"""The questions buyers ask every online shop, ready before the seller writes a word.

Each one is recognised by its usual wordings. The shop's own FAQ always wins: when an entry there covers the question,
the assistant answers from it. Otherwise:
- "how do I order / how do I pay / when are you open" are answered from the shop's own settings (order taking on or
  off, payment, hours), the same every time and with no model call;
- the rest are facts only the seller knows (shipping cost, returns, delivery time…): the assistant never guesses
  them, and the seller is told once that customers ask it, so it goes in the FAQ.
The same list drives the panel's "questions customers ask that your FAQ doesn't answer yet".
"""

import re
from dataclasses import dataclass
from typing import Any

from app.models import ChatAgent
from app.services.chat_engine import DAY_KEYS, norm, tehran_now


@dataclass(frozen=True)
class Common:
    key: str
    question: str  # how the panel shows it
    pattern: re.Pattern[str]
    hint: str  # an example answer the seller can start from
    built: bool = False  # answered from the shop's settings when the FAQ doesn't


def _re(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


_HOW = r"(چ\s*ط\s*و\s*ر\s*ی?|چجوری|چگونه|نحوه|روند|مراحل|راهنمای|چیکار\s*کنم|چکار\s*کنم|باید\s*چی\s*کار)"
COMMON: list[Common] = [
    Common("how_to_order", "چطوری سفارش بدم؟",
           _re(rf"{_HOW}\s*(ی\s*)?(باید\s*)?(سفارش|خرید|ثبت)|(سفارش|خرید)\S*\s*(رو\s*)?{_HOW}|"
               r"(از\s*کجا|کجا)\s*(باید\s*)?(سفارش|خرید)"),
           "اسم یا عکس محصول رو بفرست، رنگ و سایز رو بگو، بعد اسم و شماره و آدرس؛ خلاصه رو تأیید کن و فیش رو بفرست.",
           built=True),
    Common("payment", "چطوری پرداخت کنم؟",
           _re(rf"{_HOW}\s*(ی\s*)?(باید\s*)?(پرداخت|واریز|پول)|(پرداخت|واریز)\S*\s*(رو\s*)?{_HOW}|"
               r"(روش|راه)\s*(های|‌های)?\s*پرداخت|(کارت\s*به\s*کارت|درگاه)\s*(دارید|دارین|هست|میشه|می‌شه)"),
           "کارت‌به‌کارت؛ شماره کارت بعد از ثبت سفارش فرستاده می‌شه.", built=True),
    Common("hours", "ساعت کاری‌تون چنده؟",
           _re(r"ساعت\s*(ی\s*)?(کاری|کار)|تا\s*(چه\s*)?ساعت\s*(باز|هستید|هستین|جواب)|کی\s*(باز|بازید|بازین)|"
               r"(امروز|جمعه|فردا)\s*(باز|هستید|هستین|کار\s*می\s*‌?کنید)"),
           "شنبه تا پنجشنبه ۱۰ تا ۲۲.", built=True),
    Common("shipping_cost", "هزینهٔ ارسال چقدره؟",
           _re(r"(هزینه|خرج|کرایه|پول)\S*\s*(ی\s*)?(ارسال|پست|پیک|حمل)|(ارسال|پست|پیک)\S*\s*(چنده|چقدره|رایگان|مجانی|"
               r"هزینه)|ارسال\s*رایگان"),
           "پست پیشتاز ۸۰ هزار تومان؛ خرید بالای ۲ میلیون رایگانه."),
    Common("delivery_time", "چند روزه می‌رسه؟",
           _re(r"چند\s*روز(ه)?\s*(می\s*‌?رسه|میرسه|میاد|طول|دستم|ارسال|تحویل)|(زمان|مدت)\s*(ی\s*)?(ارسال|تحویل|رسیدن)|"
               r"کی\s*(می\s*‌?رسه|میرسه|به\s*دستم|ارسال\s*می\s*‌?کنید|میفرستید|می‌فرستید)"),
           "تهران ۱ تا ۲ روز کاری، شهرستان ۳ تا ۵ روز کاری بعد از تأیید پرداخت."),
    Common("cities", "به شهرستان هم ارسال دارید؟",
           _re(r"(شهرستان|کل\s*ایران|سراسر\s*(ایران|کشور)|شهر\s*(ما|من|دیگه|دیگر))|"
               r"به\s*\S+\s*(هم\s*)?(ارسال\s*(دارید|دارین|می\s*‌?کنید|میکنید)|میفرستید|می‌فرستید|میفرستین|می‌فرستین)"),
           "بله، به همهٔ شهرها با پست ارسال داریم."),
    Common("cod", "پرداخت در محل دارید؟",
           _re(r"(پرداخت|حساب|پول)\S*\s*(در|تو|توی)\s*محل|در\s*محل\s*(پرداخت|حساب)|موقع\s*تحویل\s*(پرداخت|حساب|پول)"),
           "نه، فقط کارت‌به‌کارت قبل از ارسال."),
    Common("returns", "امکان تعویض یا مرجوعی دارید؟",
           _re(r"(مرجوع|تعویض|پس\s*(بدم|بدیم|بفرستم|گرفتن)|برگشت\s*(بزنم|بدم|کالا|وجه)|گارانتی|ضمانت|"
               r"(اندازه|سایز)\S*\s*نشد|اگه\s*(خوب|اندازه)\s*نبود)"),
           "تا ۷ روز بعد از تحویل، اگه پوشیده نشده باشه، تعویض داریم؛ هزینهٔ ارسالش با مشتریه."),
    Common("in_person", "حضوری هم می‌شه خرید کرد؟ آدرس‌تون کجاست؟",
           _re(r"(حضوری|مغازه|شعبه|فروشگاه\s*(حضوری|فیزیکی)|آدرس\s*(تون|ت|فروشگاه|مغازه)|ادرس\s*(تون|ت|فروشگاه|مغازه)|"
               r"کجا\s*(هستید|هستین|هستی|واقع|ه\s*مغازه))"),
           "فقط آنلاین می‌فروشیم."),
    Common("discount", "تخفیف دارید؟",
           _re(r"(تخفیف|کد\s*تخفیف|حراج|جشنواره|ارزون\s*(تر)?\s*(حساب|میدی|میدید|میدین))"),
           "برای خرید اول ۱۰٪ تخفیف داریم."),
    Common("wholesale", "عمده هم می‌فروشید؟",
           _re(r"(عمده|همکاری\s*(در\s*فروش)?|نمایندگی|پخش\s*(کننده)?|تعداد\s*(بالا|زیاد))"),
           "برای خرید عمده (از ۱۰ عدد) با ادمین هماهنگ کن."),
    Common("trust", "از کجا مطمئن بشم کلاهبرداری نیست؟",
           _re(r"(کلاه\s*‌?بردار|اعتماد|مطمئن\s*(بشم|باشم)|(ای\s*)?نماد|اینماد|معتبر(ید|ین|ه)?\s*[؟?]|رضایت\s*مشتری)"),
           "نماد اعتماد داریم و رضایت مشتری‌ها توی هایلایت‌های پیجمونه."),
    Common("size_help", "چه سایزی برام مناسبه؟",
           _re(r"(چه|کدوم)\s*سایز(ی)?\s*(بردارم|بگیرم|مناسب|میخوره|می‌خوره|بخرم|بزنم)|سایز\s*(من|م)\s*(چنده|چیه)|"
               r"قد\s*(م)?\s*\d|وزن\s*(م)?\s*\d"),
           "جدول سایز رو توی «مراجع» بذار تا دستیار با اعدادش جواب بده."),
]
BY_KEY = {c.key: c for c in COMMON}


def match(text: str) -> Common | None:
    """The common question this message asks, if any (the first that fits)."""
    t = norm(text)
    return next((c for c in COMMON if c.pattern.search(t)), None)


def faq_entry(agent: ChatAgent, c: Common) -> dict[str, Any] | None:
    """The shop's FAQ entry that answers it (one asking it the usual way), if any."""
    return next((f for f in agent.faqs or [] if f.get("q") and f.get("a") and c.pattern.search(norm(str(f["q"])))),
                None)


def faq_covers(agent: ChatAgent, c: Common) -> bool:
    return faq_entry(agent, c) is not None


def gaps(agent: ChatAgent) -> list[Common]:
    """The common questions the shop's FAQ doesn't answer (the built ones only when the settings can't either)."""
    return [c for c in COMMON if not faq_covers(agent, c) and not (c.built and built_answer(agent, c.key))]


_DAY_FA = {"sat": "شنبه", "sun": "یکشنبه", "mon": "دوشنبه", "tue": "سه‌شنبه", "wed": "چهارشنبه", "thu": "پنجشنبه",
           "fri": "جمعه"}


def built_answer(agent: ChatAgent, key: str) -> str:
    """The answer the shop's settings give; "" when they don't say."""
    pay = agent.payment or {}
    has_card = bool(pay.get("card_number") or pay.get("sheba") or pay.get("link"))
    receipt = pay.get("require_receipt", True)
    if key == "how_to_order":
        if agent.orders_enabled:
            steps = ["۱) اسم محصولی که می‌خوای رو بنویس یا عکسش رو بفرست",
                     "۲) رنگ و سایزش رو انتخاب کن",
                     "۳) اسم، شماره موبایل و آدرست رو بفرست تا خلاصهٔ سفارش رو برات بفرستم",
                     "۴) با «بله» سفارشت ثبت می‌شه و کد رهگیری و "
                     + ("شماره کارت رو برات می‌فرستم" if has_card else "روش پرداخت رو بهت می‌گم")]
            if receipt:
                steps.append("۵) فیش واریزی یا کد پیگیری رو همین‌جا بفرست؛ بعد از تأیید پرداخت سفارشت آماده و ارسال "
                             "می‌شه و با همون کد رهگیری وضعیتش رو می‌تونی بپرسی")
            return "خیلی راحته 🌸\n" + "\n".join(steps) + "\nاز کدوم محصول شروع کنیم؟"
        return ("خیلی راحته 🌸\n۱) اسم محصولی که می‌خوای رو بنویس یا عکسش رو بفرست\n۲) رنگ و سایزش رو بگو\n"
                "۳) بعد همکارم برای ثبت سفارش و پرداخت بهت پیام می‌ده\nاز کدوم محصول شروع کنیم؟")
    if key == "payment":
        if not agent.orders_enabled:
            return ""
        if pay.get("link") and not pay.get("card_number"):
            how = "با لینک پرداختی که بعد از ثبت سفارش می‌فرستم"
        elif has_card:
            how = "کارت‌به‌کارت؛ شماره کارت رو بعد از ثبت سفارش (وقتی خلاصه رو تأیید کردی) برات می‌فرستم"
        else:
            return ""
        return f"پرداخت {how}" + ("، و فیش یا کد پیگیریش رو همین‌جا می‌فرستی 🙏" if receipt else " 🙏")
    if key == "hours":
        hours = agent.business_hours or {}
        if not hours.get("enabled"):
            return ""
        days = hours.get("days") or {}
        today = DAY_KEYS[tehran_now().weekday()]
        rows = [f"{_DAY_FA[k]}: " + ("، ".join(f"{a} تا {b}" for a, b in days.get(k) or []) or "تعطیل") for k in _DAY_FA]
        now = "، ".join(f"{a} تا {b}" for a, b in days.get(today) or []) or "تعطیلیم"
        return f"امروز ({_DAY_FA[today]}) {now} 🌸\n" + "\n".join(rows) + "\nپیام بذاری، توی ساعت کاری جواب می‌دیم."
    return ""


def gap_note(c: Common, text: str) -> str:
    """The seller's note when a customer asks what the FAQ doesn't answer."""
    return (f"❓ مشتری پرسید «{text[:100]}» ({c.question}) و جوابش توی «سوالات متداول» دستیار نیست. اضافه‌اش کن تا "
            "دستیار خودش جواب بده.")


def suggestions(agent: ChatAgent) -> list[dict[str, Any]]:
    return [{"key": c.key, "q": c.question, "hint": c.hint} for c in gaps(agent)]
