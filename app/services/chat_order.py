"""The order journey, in one place: what the cart still needs and what the shop says next.

    cart → choice (size, colour, print side…) → design (print products) → details (name, phone, address, extras)
         → summary («بله» / «تغییر می‌خوام») → placed → receipt

The model only reads the customer's words into the cart (which product, which choices, the contact details) and
writes the chat text; whether the summary shows, what is asked next and when an order is placed is decided here,
for typed messages and button taps alike."""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import utcnow
from app.models import AiOrder, ChatAgent, ChatConversation, Product
from app.services import chat_engine as ce
from app.services.text import fa_number

YES_BUTTON = "✅ بله، ثبت کن"
CHANGE_BUTTON = "✏️ تغییر می‌خوام"
SUMMARY_BUTTONS = [[YES_BUTTON, CHANGE_BUTTON]]
SUMMARY_ASK = ("همه‌چی درسته؟ اگه درسته «بله» یا «آره» بفرست، وگرنه بگو چی رو عوض کنم "
               "(یا اگه چیز دیگه‌ای هم می‌خوای، بگو تا اضافه‌ش کنم).")
ASK_FIELD = {"name": "اسم و فامیلت رو بنویس 👤", "phone": "شماره موبایلت رو بفرست 📞",
             "address": "آدرس کامل رو با کد پستی بنویس 📍"}
SEND_PHONE = "📱 ارسال شماره‌ی من"
_KIND_NAMES = {"color": "رنگ", "size": "سایز"}
MORE_YES = "➕ آره، یه چیز دیگه"
MORE_NO = "✅ نه، ادامه‌ی ثبت سفارش"
MORE_BUTTONS = [[MORE_YES, MORE_NO]]
MORE_ASK = "چیز دیگه‌ای هم می‌خوای به سفارشت اضافه کنی؟ 🌸"
MORE_WHAT = "باشه 🌸 اسم یا عکس محصول بعدی رو بفرست تا اضافه‌ش کنم."
MORE_DONE = re.compile(r"^(نه|نخیر|خیر|نچ|no|nope)?\s*(ممنون|مرسی|کافیه|بسه|همین(ه|ا|ها|قدر)?|تموم(ه)?|نه|نمی\s*‌?خوام|"
                       r"نیست|نه\s*دیگه|ادامه|ثبت\s*کن|بریم)?\s*[.!🌸🙏]*$", re.I)


@dataclass
class Step:
    """The next thing the shop says: `stage` is choice | design | more | details | summary."""
    stage: str
    text: str
    buttons: list[list[str]] | None = None


def kind_label(kind: str) -> str:
    return _KIND_NAMES.get(kind, kind)


def choices(p: Product) -> dict[str, list[str]]:
    """The product's own options with more than one choice: colour, size, print side…"""
    return {k: v for k, v in ce.variant_options(p).items() if len(v) > 1}


def current(kind: str, options: list[str], variant: str) -> str:
    return next((o for o in options if ce._option_chosen(kind, [o], variant)), "")  # noqa: SLF001


def rows_of(options: list[str], per: int = 3) -> list[list[str]]:
    return [options[i:i + per] for i in range(0, len(options), per)]


async def product(db: AsyncSession, store_id: int, item: dict[str, Any]) -> Product | None:
    p = await db.get(Product, item.get("product_id") or 0) if item.get("product_id") else None
    return p if p and p.store_id == store_id else None


def choice_step(state: dict[str, Any], idx: int, p: Product, kind: str, options: list[str]) -> Step:
    """A missing option of the product, as buttons; a tap (or the typed word) is written on that line."""
    state["menu"] = {"step": "set_choice", "line": idx, "kind": kind}
    return Step("choice", f"برای «{p.title[:60]}» {kind_label(kind)} رو هم انتخاب کن: {'، '.join(options)} 🌸",
                rows_of(options))


async def carry_over(db: AsyncSession, conv: ChatConversation, order: dict[str, Any],
                     placed: AiOrder | None) -> None:
    """Contact details (and the placed order's extras) from an earlier order: nothing is asked twice."""
    earlier = placed or await ce.last_order(db, conv)
    if not earlier:
        return
    for k, v in (("name", earlier.customer_name), ("phone", earlier.customer_phone),
                 ("address", earlier.customer_address)):
        if v and not order.get(k):
            order[k] = v
    if placed and placed.extras:
        order["extras"] = {**placed.extras, **(order.get("extras") or {})}


async def cart_complete(db: AsyncSession, store_id: int, items: list[dict[str, Any]]) -> bool:
    return await ce.items_complete(db, store_id, items)


async def next_step(db: AsyncSession, conv: ChatConversation, agent: ChatAgent, state: dict[str, Any],
                    placed: AiOrder | None, one_by_one: bool = False) -> Step | None:
    """The first thing the cart still needs, else the summary (and the cart then waits for «بله»).
    None when the cart is empty. `one_by_one`: details asked one field at a time (the button flow)."""
    order = dict(state.get("order") or {})
    items = order.get("items") or []
    if not items:
        return None
    store_id = conv.store_id
    for idx, it in enumerate(items):
        p = await product(db, store_id, it)
        if not p:
            continue
        variant = str(it.get("variant") or "")
        for kind, options in choices(p).items():
            if not current(kind, options, variant):
                return choice_step(state, idx, p, kind, options)
    for it in items:
        p = await product(db, store_id, it)
        if p and ce.takes_design(p) and not it.get("design"):
            state.pop("menu", None)
            return Step("design", f"برای «{p.title[:60]}» طرحی که می‌خوای چاپ بشه رو همین‌جا بفرست (عکس یا فایل) 🎨")
    await carry_over(db, conv, order, placed)
    state["order"] = order
    missing = ce._missing_fields(agent, order)  # noqa: SLF001
    if (missing and len(items) > int(state.get("more_asked") or 0) and not state.get("awaiting")
            and not any(order.get(k) for k in ("name", "phone", "address"))):
        # The cart is ready and the checkout comes next: "anything else?" first — once, again only after a new line.
        # (With the details already given, the summary comes straight away and says the same.)
        state["more_asked"] = len(items)
        state["menu"] = {"step": "more"}
        lines, total = await ce.priced_lines(db, store_id, items)
        rows = "\n".join(ce.line_text(i) for i in lines)
        return Step("more", f"🛒 سبد خریدت:\n{rows}\n💰 جمع: {ce.money(total)}\n\n{MORE_ASK}", MORE_BUTTONS)
    if missing:
        form = str((agent.payment or {}).get("order_form") or "").strip()
        contact = [m for m in missing if m in ASK_FIELD]
        if form and (len(contact) > 1 or not one_by_one) and contact:
            state.pop("menu", None)
            state["form_sent"] = True
            return Step("details", form[:1000])
        if one_by_one and missing[0] in ASK_FIELD:
            state["menu"] = {"step": "field", "field": missing[0], "then": "next"}
            return Step("details", ASK_FIELD[missing[0]], [[SEND_PHONE]] if missing[0] == "phone" else None)
        state.pop("menu", None)
        return Step("details", ce.ask_missing(agent, state, missing, thanks=False))
    state.pop("menu", None)
    lines, total = await ce.priced_lines(db, store_id, items)
    if not lines:
        return None
    state["awaiting"] = "confirm"
    state["summary_at"] = utcnow().isoformat()  # a «بله» tapped before this was for an older summary
    if placed:
        state["add_to"] = placed.id
        rows = "\n".join(ce.line_text(i) for i in lines)
        text = f"اضافه به سفارش #{fa_number(placed.id)}:\n{rows}\n💰 مبلغ این اضافه: {ce.money(total)}"
    else:
        state.pop("add_to", None)
        text = "خلاصهٔ سفارشت:\n" + ce.order_summary(order, lines, total)
    return Step("summary", f"{text}\n\n{SUMMARY_ASK}", SUMMARY_BUTTONS)
