"""The router: each message is first sorted into the part of the journey it belongs to (one short, cheap model call),
then that part's "skill" answers it. Skills the shop can answer by itself (browsing the catalog, an order's status,
the yes to the summary, the cart, a person, off-topic…) are answered by code; the rest (a product question, choosing,
the details) go to the sales model with the skill named. When the router is unsure or fails, the sales model answers
as before, so nothing is lost."""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatAgent, ChatConversation, ChatMessage, Store
from app.services import ai

log = logging.getLogger("sello.chat")

SKILLS = {
    "browse": "what kinds/models/designs do you have, other designs of X, a list (\"دیگه چه مدلایی داری؟\")",
    "product": "a question about a product: price, photo, size, size chart, material, stock, similar items",
    "cart": "choosing or changing what to buy: a product, colour, size, quantity, print side, removing an item",
    "design": "about their own design/picture to print",
    "checkout": "wants to finish the order now, with nothing new in the message (\"ثبتش کن\", \"تمومه بفرست\")",
    "details": "gives their name, phone, address or postal code",
    "confirm": "a plain yes to the order summary (only when STEP is summary)",
    "cart_view": "wants to see what is in their cart now (not a placed order)",
    "status": "the status or items of an order ALREADY placed, tracking, delivery time of it",
    "payment": "paying: card number, sent the money, the receipt",
    "faq": "shipping, returns, hours, payment ways, the shop itself",
    "greeting": "hello / thanks / bye only",
    "human": "wants a real person / the admin",
    "complaint": "a complaint about an order, product, delivery or behaviour",
    "off_topic": "anything unrelated to this shop",
    "abuse": "insults or swearing",
    "unclear": "you can't tell",
}
# Answered by the shop's code; every other skill goes to the sales model.
CODE_SKILLS = {"browse", "status", "confirm", "cart_view", "checkout", "human", "complaint", "off_topic", "abuse"}

ROUTER_SYSTEM = ("You sort ONE customer message in an Iranian online shop's chat into the part of the shopping journey it "
                 "belongs to. Read RECENT CHAT and STEP for context.\nSKILLS:\n"
                 + "\n".join(f"- {k}: {v}" for k, v in SKILLS.items())
                 + '\nReply with ONE JSON object: {"skill":"<one of SKILLS>","sure":true|false}. '
                   "Several asks in one message -> the main one. Not sure -> \"unclear\".")


async def route(db: AsyncSession, conv: ChatConversation, store: Store, agent: ChatAgent, state: dict[str, Any],
                text: str) -> str:
    """The skill for this message ("" when the router can't tell or fails: the sales model answers as before)."""
    history = (await db.execute(select(ChatMessage).where(
        ChatMessage.conversation_id == conv.id, ChatMessage.role.in_(["customer", "ai", "seller"]))
        .order_by(ChatMessage.id.desc()).limit(7))).scalars().all()
    recent = "\n".join(f"{'customer' if m.role == 'customer' else 'shop'}: {(m.text or '[photo]')[:160]}"
                       for m in reversed(history[1:]))
    cart = (state.get("order") or {}).get("items") or []
    step = {"confirm": "summary (the shop showed the order summary and waits for yes)",
            "receipt": "receipt (an order is placed and waits for payment)"}.get(str(state.get("awaiting") or ""), "browsing")
    prompt = (f"STEP: {step}\nCART: {len(cart)} item(s)\nRECENT CHAT:\n{recent or '-'}\n\n"
              f"<customer>{text[:600]}</customer>")
    try:
        out = await ai.generate_json(task="chat_router", system=ROUTER_SYSTEM, prompt=prompt, store_id=store.id,
                                     max_tokens=40)
    except Exception:  # noqa: BLE001 - no router model or a bad answer: the sales model answers as before
        log.info("chat router unavailable", exc_info=True)
        return ""
    skill = str(out.get("skill") or "").strip()
    if skill not in SKILLS or out.get("sure") is False:
        return ""
    return skill

