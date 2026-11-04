"""The cart and order buttons (bot mode): «🛒 سبد خرید من», «✅ ثبت سفارش», the summary's yes / change, and each
step of changing the cart. Taps are answered by code, never by the model, with the texts agreed with the seller;
anything typed goes on to the usual model path, so the journey stays the same for customers who type."""

import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.text import fa_number, to_en_digits
from app.models import ChatAgent, ChatConversation, Product, Store
from app.services import chat_engine as ce
from app.services.chat_engine import Reply

CART = "🛒 سبد خرید من"
ORDER = "✅ ثبت سفارش"
MAIN_BUTTONS = [[CART, ORDER]]
YES = "✅ بله، ثبت کن"
CHANGE = "✏️ تغییر می‌خوام"
SUMMARY_BUTTONS = [[YES, CHANGE]]
EDIT_CART = "✏️ تغییر سبد"
CLEAR = "🗑 خالی کردن سبد"
DETAILS = "📦 جزئیات سفارشم"
CH_OPTS, CH_QTY, CH_DEL = "🎨 گزینه‌های محصول", "🔢 تعداد", "➖ حذف یه آیتم"
CH_CONTACT, CH_CANCEL, BACK = "👤 اسم، شماره یا آدرس", "❌ لغو سفارش", "↩️ برگشت به خلاصه"
YES_CLEAR, YES_CANCEL, NO_BACK = "بله، خالی کن", "بله، لغو کن", "نه، برگرد"
NAME, PHONE, ADDRESS = "اسم", "شماره", "آدرس"
SEND_PHONE = "📱 ارسال شماره‌ی من"
QTY_BUTTONS = [["۱", "۲", "۳", "۴", "۵"]]
DONE = "عوض شد ✅"
EMPTY = "سبد خریدت خالیه 🌸 اسم محصولی که می‌خوای رو بنویس یا عکسش رو بفرست تا برات بذارمش توی سبد."
EMPTY_ORDER = "سبدت هنوز خالیه 🌸 اول بگو چی می‌خوای (اسم یا عکس محصول) تا برات بذارمش توی سبد."
CLEARED = "سبدت خالی شد 🌸 هر وقت خواستی، اسم محصول رو بنویس."
CANCELLED = "سفارش لغو شد 🌸 هر وقت خواستی، دوباره از «🛒 سبد خرید من» شروع کن."
ASK_FIELD = {"name": "اسم و فامیلت رو بنویس 👤", "phone": "شماره موبایلت رو بفرست 📞",
             "address": "آدرس کامل رو با کد پستی بنویس 📍"}
NEW_FIELD = {"name": "اسم جدید رو بنویس 👤", "phone": "شماره‌ی جدید رو بفرست 📞",
             "address": "آدرس جدید رو با کد پستی بنویس 📍"}
_KIND_NAMES = {"color": "رنگ", "size": "سایز"}
_LINE_NO = re.compile(r"^\s*(\d+)\)")


def _r(text: str, buttons: list[list[str]] | None = None) -> Reply:
    return Reply(text, source="menu", intent="order", buttons=buttons)


def _label(kind: str) -> str:
    return _KIND_NAMES.get(kind, kind)


def _choices(p: Product) -> dict[str, list[str]]:
    """The product's own options with more than one choice: colour, size, print side…"""
    return {k: v for k, v in ce.variant_options(p).items() if len(v) > 1}


def _current(kind: str, options: list[str], variant: str) -> str:
    return next((o for o in options if ce._option_chosen(kind, [o], variant)), "")  # noqa: SLF001


def set_choice(p: Product, variant: str, kind: str, new: str) -> str:
    """The variant with one choice replaced ("مشکی 2XL جلو" + پشت → "مشکی 2XL پشت"); the rest stays."""
    cur = _current(kind, ce.variant_options(p).get(kind, []), variant)
    if cur and re.search(re.escape(cur), variant, flags=re.I):
        return re.sub(re.escape(cur), new, variant, count=1, flags=re.I)[:80]  # in place, the order kept
    v = variant
    for o in sorted(ce.variant_options(p).get(kind, []), key=len, reverse=True):
        v = re.sub(re.escape(o), " ", v, flags=re.I)
    if kind == "color":
        for c in ce._colors_in(v):  # noqa: SLF001
            v = v.replace(c, " ")
    elif kind == "size":
        v = to_en_digits(v)
        for w in ce._SIZE_WORDS:  # noqa: SLF001
            v = v.replace(w, " ")
        v = ce._SIZE_TOKEN.sub(" ", v)  # noqa: SLF001
    return " ".join(f"{v} {new}".split())[:80]


def _line_buttons(lines: list[dict[str, Any]]) -> list[list[str]]:
    return [[f"{fa_number(n)}) {i['title'][:40]}" + (f" ({i['variant']})" if i.get("variant") else "")]
            for n, i in enumerate(lines, 1)]


def _line_no(text: str, count: int) -> int | None:
    m = _LINE_NO.match(to_en_digits(text))
    n = int(m.group(1)) if m else 0
    return n - 1 if 1 <= n <= count else None


async def _product(db: AsyncSession, store_id: int, item: dict[str, Any]) -> Product | None:
    p = await db.get(Product, item.get("product_id") or 0) if item.get("product_id") else None
    return p if p and p.store_id == store_id else None


async def cart_text(db: AsyncSession, store_id: int, items: list[dict[str, Any]]) -> str:
    lines, total = await ce.priced_lines(db, store_id, items)
    rows = [f"{fa_number(n)}) " + ce.line_text(i).removeprefix("▫️ ") for n, i in enumerate(lines, 1)]
    return "🛒 سبد خریدت:\n" + "\n".join(rows) + f"\n💰 جمع: {ce.money(total)}"


async def summary(db: AsyncSession, store_id: int, state: dict[str, Any], lead: str = "") -> Reply | None:
    order = state.get("order") or {}
    lines, total = await ce.priced_lines(db, store_id, order.get("items") or [])
    if not lines:
        return None
    state["awaiting"] = "confirm"
    state.pop("add_to", None)
    text = "خلاصهٔ سفارشت:\n" + ce.order_summary(order, lines, total) + "\n\nهمه‌چی درسته؟"
    return _r((lead + "\n\n" if lead else "") + text, SUMMARY_BUTTONS)


def ask_choice(state: dict[str, Any], idx: int, p: Product, kind: str, options: list[str], variant: str,
               missing: bool) -> Reply:
    """One of the product's own options as buttons; the tap is written on that line."""
    state["menu"] = {"step": "set_choice", "line": idx, "kind": kind}
    cur = _current(kind, options, variant)
    if missing:
        return _r(f"برای «{p.title[:60]}» {_label(kind)} رو انتخاب کن:", [options[i:i + 3] for i in range(0, len(options), 3)])
    marked = [("✔️ " + o if o == cur else o) for o in options]
    return _r(f"{_label(kind)} رو انتخاب کن" + (f" (الان: {cur})" if cur else "") + ":",
              [marked[i:i + 3] for i in range(0, len(marked), 3)])


async def next_step(db: AsyncSession, conv: ChatConversation, store: Store, agent: ChatAgent,
                    state: dict[str, Any], lead: str = "") -> list[Reply] | None:
    replies = await _next_step(db, conv, store, agent, state, lead)
    if replies and lead and not replies[0].text.startswith(lead):
        replies[0].text = f"{lead}\n\n{replies[0].text}"  # "عوض شد ✅" before whatever comes next
    return replies


async def _next_step(db: AsyncSession, conv: ChatConversation, store: Store, agent: ChatAgent,
                     state: dict[str, Any], lead: str = "") -> list[Reply] | None:
    """«ثبت سفارش»: the first thing still missing (a choice, the design, a detail), else the summary. None: the
    model path takes it (an addition to an order already placed has its own flow there)."""
    order = dict(state.get("order") or {})
    items = order.get("items") or []
    if not items:
        placed = await ce.open_order(db, store.id, state.get("order_id"))
        if placed and placed.status == "awaiting_payment":
            state["awaiting"] = "receipt"
            return [_r(f"سفارش #{fa_number(placed.id)} منتظر پرداخته 🌸\n" + ce.payment_text(agent, placed))]
        return [_r(EMPTY_ORDER)]
    if await ce.open_order(db, store.id, state.get("order_id")):
        return None
    for idx, it in enumerate(items):
        p = await _product(db, store.id, it)
        if not p:
            continue
        variant = str(it.get("variant") or "")
        for kind, options in _choices(p).items():
            if not _current(kind, options, variant):
                return [ask_choice(state, idx, p, kind, options, variant, missing=True)]
        if ce.takes_design(p) and not it.get("design"):
            state.pop("menu", None)
            return [_r(f"برای «{p.title[:60]}» طرحی که می‌خوای چاپ بشه رو همین‌جا بفرست (عکس یا فایل) 🎨")]
    earlier = await ce.last_order(db, conv)
    if earlier:
        for k, v in (("name", earlier.customer_name), ("phone", earlier.customer_phone),
                     ("address", earlier.customer_address)):
            if v and not order.get(k):
                order[k] = v
        state["order"] = order
    missing = ce._missing_fields(agent, order)  # noqa: SLF001
    if missing:
        form = str((agent.payment or {}).get("order_form") or "").strip()
        if form and len([m for m in missing if m in ASK_FIELD]) > 1:
            state.pop("menu", None)
            state["form_sent"] = True
            return [_r(form[:1000])]
        field_ = missing[0]
        if field_ in ASK_FIELD:
            state["menu"] = {"step": "field", "field": field_, "then": "next"}
            return [_r(ASK_FIELD[field_], [[SEND_PHONE]] if field_ == "phone" else None)]
        state.pop("menu", None)
        return [_r(ce.ask_missing(agent, state, missing, thanks=False))]
    state.pop("menu", None)
    r = await summary(db, store.id, state, lead)
    return [r] if r else [_r(EMPTY_ORDER)]


async def _change_menu(state: dict[str, Any]) -> list[Reply]:
    state["menu"] = {"step": "change"}
    return [_r("چی رو عوض کنم؟", [[CH_OPTS, CH_QTY], [CH_DEL, CH_CONTACT], [CH_CANCEL, BACK]])]


async def _for_line(db: AsyncSession, store: Store, state: dict[str, Any], idx: int, then: str) -> list[Reply]:
    items = (state.get("order") or {}).get("items") or []
    it = items[idx]
    p = await _product(db, store.id, it)
    title = (p.title if p else str(it.get("title") or ""))[:60]
    if then == "opts":
        choices = _choices(p) if p else {}
        if not choices:
            state.pop("menu", None)
            return [_r(f"«{title}» گزینه‌ای برای انتخاب نداره 🌸 اگه چیزی رو می‌خوای عوض کنی، بنویس.")]
        if len(choices) == 1:
            kind, options = next(iter(choices.items()))
            return [ask_choice(state, idx, p, kind, options, str(it.get("variant") or ""), missing=False)]
        state["menu"] = {"step": "pick_kind", "line": idx}
        return [_r(f"برای «{title}» کدوم رو عوض کنم؟", [[_label(k) for k in choices]])]
    if then == "qty":
        state["menu"] = {"step": "qty", "line": idx}
        return [_r(f"چند تا «{title}» می‌خوای؟", QTY_BUTTONS)]
    # remove the line
    del items[idx]
    state["order"] = {**(state.get("order") or {}), "items": items}
    state.pop("menu", None)
    state.pop("awaiting", None)
    head = f"«{title}» از سبد حذف شد ✅"
    if not items:
        return [_r(f"{head}\n{CLEARED}")]
    r = await summary(db, store.id, state, head)
    return [r] if r else [_r(head)]


async def handle(db: AsyncSession, conv: ChatConversation, store: Store, agent: ChatAgent, state: dict[str, Any],
                 text: str, result: ce.TurnResult) -> list[Reply] | None:
    """The answer to a button tap (or to the one thing a button step asked for); None: the model path answers."""
    if not agent.orders_enabled:
        return None
    t = " ".join(text.split())
    menu = dict(state.get("menu") or {})
    step = menu.get("step", "")
    order = dict(state.get("order") or {})
    items = list(order.get("items") or [])

    if t == CART:
        state.pop("menu", None)
        if items:
            return [_r(await cart_text(db, store.id, items), [[ORDER], [EDIT_CART, CLEAR]])]
        last = await ce.last_order(db, conv)
        if last and last.status in ce.OPEN_ORDER:
            return [_r(f"{EMPTY}\nآخرین سفارشت: #{fa_number(last.id)}", [[DETAILS]])]
        return [_r(EMPTY)]
    if t == ORDER:
        state.pop("menu", None)
        return await next_step(db, conv, store, agent, state)
    if t == DETAILS:
        state.pop("menu", None)
        last = await ce.last_order(db, conv)
        return [_r(ce.order_status_text(last) if last else "هنوز سفارشی ثبت نکردی 🌸")]
    if t == YES and state.get("awaiting") == "confirm" and items:
        state.pop("menu", None)
        state.pop("awaiting", None)
        placed = ce.TurnResult()
        await ce.place_order(db, conv, store, agent, state, t, "", placed)
        result.order_id = placed.order_id
        result.notify_admin += placed.notify_admin
        return placed.replies
    if t in (CHANGE, EDIT_CART):
        if not items:
            state.pop("menu", None)
            return [_r(EMPTY)]
        return await _change_menu(state)
    if t == CLEAR:
        if not items:
            return [_r(EMPTY)]
        state["menu"] = {"step": "clear"}
        return [_r("مطمئنی همهٔ آیتم‌ها از سبد پاک بشن؟", [[YES_CLEAR, NO_BACK]])]
    if t in (BACK, NO_BACK):
        state.pop("menu", None)
        return await next_step(db, conv, store, agent, state)
    if t == YES_CLEAR and step == "clear":
        for k in ("order", "awaiting", "menu", "add_to", "form_sent"):
            state.pop(k, None)
        return [_r(CLEARED)]
    if t == YES_CANCEL and step == "cancel":
        for k in ("order", "awaiting", "menu", "add_to", "form_sent"):
            state.pop(k, None)
        return [_r(CANCELLED)]

    if not step or not items and step != "field":
        return None
    if step == "change":
        if t == CH_CANCEL:
            state["menu"] = {"step": "cancel"}
            return [_r("مطمئنی کل سفارش لغو بشه؟", [[YES_CANCEL, NO_BACK]])]
        if t == CH_CONTACT:
            state["menu"] = {"step": "contact"}
            return [_r("کدوم رو عوض کنم؟", [[NAME, PHONE, ADDRESS]])]
        then = {CH_OPTS: "opts", CH_QTY: "qty", CH_DEL: "del"}.get(t)
        if then:
            if len(items) > 1:
                lines, _ = await ce.priced_lines(db, store.id, items)
                state["menu"] = {"step": "pick_line", "then": then}
                return [_r("کدوم آیتم؟", _line_buttons(lines))]
            return await _for_line(db, store, state, 0, then)
    elif step == "pick_line":
        idx = _line_no(t, len(items))
        if idx is not None:
            return await _for_line(db, store, state, idx, menu.get("then", "opts"))
    elif step == "pick_kind":
        idx = int(menu.get("line") or 0)
        p = await _product(db, store.id, items[idx]) if idx < len(items) else None
        kind = next((k for k in (_choices(p) if p else {}) if _label(k) == t), None)
        if p and kind:
            return [ask_choice(state, idx, p, kind, _choices(p)[kind], str(items[idx].get("variant") or ""),
                               missing=False)]
    elif step == "set_choice":
        idx, kind = int(menu.get("line") or 0), str(menu.get("kind") or "")
        p = await _product(db, store.id, items[idx]) if idx < len(items) else None
        options = ce.variant_options(p).get(kind, []) if p else []
        pick = t.removeprefix("✔️").strip()
        if p and pick in options:
            items[idx] = {**items[idx], "variant": set_choice(p, str(items[idx].get("variant") or ""), kind, pick)}
            state["order"] = {**order, "items": items}
            state.pop("menu", None)
            return await next_step(db, conv, store, agent, state, lead=DONE)
    elif step == "qty":
        idx = int(menu.get("line") or 0)
        n = int(to_en_digits(t)) if to_en_digits(t).isdigit() else 0
        if 1 <= n <= 99 and idx < len(items):
            p = await _product(db, store.id, items[idx])
            if p and 0 < p.stock_qty < n and ce.policy(agent)["stock"] == "exact":
                return [_r(f"از «{p.title[:60]}» فقط {fa_number(p.stock_qty)} تا داریم 🙏 چند تا بذارم؟", QTY_BUTTONS)]
            items[idx] = {**items[idx], "qty": n}
            state["order"] = {**order, "items": items}
            state.pop("menu", None)
            return await next_step(db, conv, store, agent, state, lead=DONE)
    elif step == "contact":
        field_ = {NAME: "name", PHONE: "phone", ADDRESS: "address"}.get(t)
        if field_:
            state["menu"] = {"step": "field", "field": field_, "then": "summary"}
            return [_r(NEW_FIELD[field_], [[SEND_PHONE]] if field_ == "phone" else None)]
    elif step == "field":
        field_ = str(menu.get("field") or "")
        value = text.strip()
        if field_ == "phone":
            phone = ce.norm_phone(value)
            if not phone:
                bad = ce.bad_phone(value)
                if bad or not re.sub(r"[\d\s+-]", "", to_en_digits(value)):
                    return [_r("این شماره درست نیست 🙏 شماره موبایل ۱۱ رقمیه و با ۰۹ شروع میشه. دوباره بفرست:",
                               [[SEND_PHONE]])]
                state.pop("menu", None)
                return None  # they wrote something else: the model reads it
            value = phone
        elif field_ == "name" and (ce.norm_phone(value) or len(value) > 60 or "\n" in value):
            state.pop("menu", None)
            return None  # name, phone and address in one message: the model sorts them out
        elif field_ == "address" and len(value) < 5:
            return [_r(ASK_FIELD["address"])]
        if field_ in ASK_FIELD and value:
            state["order"] = {**order, field_: value[:500]}
            state.pop("menu", None)
            lead = DONE if menu.get("then") == "summary" else ""
            return await next_step(db, conv, store, agent, state, lead=lead)
    # Anything else (typed text): the step is left and the model answers as usual.
    state.pop("menu", None)
    return None

