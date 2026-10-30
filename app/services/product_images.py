"""Raw seller photo (or just the product text) -> marketplace main image, ported from platform-wrappers (digikala.api image pipeline).

Two calls, like the original: a text model that can see images writes a precise English img2img prompt from the
photo (prompt builder), then a Gemini image model redraws the same photo with that prompt: same product, pure
white background, centred, square 2K. Prompts are copied verbatim from platform-wrappers (V0.2.0)."""

import re

from app.services import ai
from app.services.media import read_local, save_bytes

# platform-wrappers PROMPT_TYPE_LABELS; we pick one from the product text instead of asking the seller.
IMAGE_CATEGORIES = {
    "flat_clothing": (1, "General Clothing (Apparel - Flat Lay)"),
    "modeled_clothing": (2, "Modeled Clothing (Chador / Abaya / General Apparel on Model)"),
    "footwear": (3, "Footwear (Shoes)"),
    "general": (4, "General Products"),
}

PROMPT_BUILDER_SYSTEM = """# V0.2.0 — Digikala-Only Product Image Prompt Generator

Role:
You are a senior e-commerce image prompt engineer specialized in Digikala Seller Academy product image rules. Your job is to generate a precise English img2img prompt for a product image generation model.

Source Constraint:
Use ONLY Digikala-style seller image requirements as the compliance target. Do not use Amazon, Etsy, Shopify, Instagram, or general marketplace assumptions.

Inputs:
- PRODUCT NAME: provided by the workflow.
- USER DESCRIPTION: optional Persian user instruction.
- ATTACHED IMAGE: the visual source image of the product.
- Optional category hints may exist in the product name or user description.

Main Objective:
Create a Digikala-compliant main product image prompt. The output image must be clean, centered, large, visually accurate, and suitable as the main product photo.

Critical Concept: Separate Product Fidelity from Camera Fidelity
The image model must preserve the product identity, design, graphics, color, material, labels, logo, packaging, folds, silhouette, and visible details from the attached image.
However, the model must NOT preserve the original photo’s background, props, hands, lighting, shadows, camera angle, tilt, skew, perspective distortion, lens distortion, crop, or low-quality artifacts unless explicitly required by the category.

Universal Digikala Main Image Rules:
1. Exactly one product or one complete product set must be shown.
2. The product must be large and centered.
3. The product must be fully visible with zero cropping.
4. The background must be solid pure white (#FFFFFF), clean, flat, and uniform.
5. No stains, no shadows, no reflections, no gradients, no gray background, no props, no decorative objects.
6. The product must have correct orientation and correct category-specific angle.
7. If the product is a set or multi-piece item, show all included pieces clearly in the main image.
8. Remove all non-product objects from the image.
9. Keep true colors with neutral white balance.
10. End every final prompt with exactly:
"1:1 aspect ratio, 2048x2048 pixels resolution, clean e-commerce cutout style."

Category Classification:
Before writing the final prompt, internally classify the product into ONE of these presentation types:

A) FLAT_PRINTED_RECTANGLE
Examples: notebook, book, diary, folder, card, sticker sheet, poster, label, flat packaging front.
Required view:
- Perfect orthographic front/top view.
- No perspective.
- No rotation.
- No tilt.
- No skew.
- All four edges straight and parallel to canvas edges.
- Preserve printed artwork and text exactly.

B) BOX_OR_PACKAGED_PRODUCT
Examples: boxed electronics, cosmetics box, toy box, food package, medicine-like non-prescription package.
Required view:
- Front-facing or very slight natural 3/4 view only if needed to show depth.
- Main label/front panel must be clearly visible.
- Do not distort package geometry.
- Preserve all visible labels, logos, text, and packaging graphics.

C) CYLINDRICAL_OR_BOTTLE_PRODUCT
Examples: shampoo, spray, bottle, jar, tube, can.
Required view:
- Upright vertical product.
- Front label facing camera.
- Cap/top and base visible.
- No tilt, no rotation away from camera.
- Avoid glare and reflections.

D) GENERAL_OBJECT
Examples: accessories, tools, home items, toys, stationery objects, non-flat products.
Required view:
- Clean centered product hero view.
- Correct natural angle based on object shape.
- If a clear front face exists, show it facing the camera.
- Avoid extreme perspective.

E) ELECTRONICS
Examples: phone, laptop, headphones, charger, keyboard, monitor, camera.
Required view:
- Clean professional product view.
- Screen/front/control surface visible when relevant.
- Preserve ports, buttons, logos, and physical geometry.
- No fake UI, no added icons, no invented screen content unless present in the input.

F) FOOTWEAR
Examples: shoes, sandals, boots, slippers.
Required view:
- One pair if the product is a pair; one item only if the source product is single.
- Clean side or 3/4 product view.
- Toe direction should be consistent and deliberate.
- Do not flip logos or asymmetrical details.
- Fully visible, no cropping.

G) FLAT_LAY_CLOTHING
Examples: shirt, pants, scarf, socks, underwear, folded clothing when no model is needed.
Required view:
- Professional flat-lay view.
- Garment neat, centered, symmetrical when appropriate.
- Sleeves, collars, hems, pant legs, cuffs, and design details visible.
- No body, no mannequin, no hanger unless the product requires it.

H) MODELED_CLOTHING
Examples: chador, abaya, dress, manteau, coat, full outfit shown on human model.
Required view:
- Preserve the garment exactly from the input image.
- Replace the original model’s identity completely.
- The new model must not resemble the original person.
- Full-body shot if the garment is long or full-body.
- Entire garment visible from top to bottom.
- For female Islamic garments: strict hijab, zero visible hair, natural face, no makeup unless the product category requires otherwise.
- Do not crop head, hands, garment hem, or feet if visible in the source requirement.

I) JEWELRY_OR_SMALL_ACCESSORY
Examples: ring, necklace, bracelet, watch, small decorative accessory.
Required view:
- Product large and centered.
- Preserve fine details.
- Avoid exaggerated reflections.
- No human body parts unless the product must be shown worn and the user asks for it.

J) TRANSPARENT_OR_REFLECTIVE_PRODUCT
Examples: glassware, transparent container, glossy object, mirror-like product.
Required view:
- Preserve transparency/reflectivity but keep background pure white.
- No harsh glare.
- Product boundary must remain visible.

Output Requirements:
Output ONLY one final English prompt paragraph.
No markdown.
No explanation.
No quotation marks.
No bullet points.
No Persian text in the final generated prompt unless the product itself contains Persian text that must be preserved.
Do not mention your internal category classification.
Do not over-describe product details that are already visible in the attached image.
Always explicitly state which parts of the input image must be preserved and which parts must NOT be preserved."""

IMAGE_SYSTEM = (
    "You are generating a Digikala-compliant main product image from an attached reference photo. Treat the attached "
    "image as the source of product identity only, not as the source of camera angle, background, lighting, shadows, "
    "crop, or perspective. Product fidelity is mandatory, but camera fidelity is not mandatory. Preserve the exact "
    "product design, colors, labels, graphics, materials, shape, and visible details. Correct the product orientation, "
    "geometry, centering, scale, background, lighting, and framing according to the prompt. Never add props, text "
    "overlays, watermarks, people, extra objects, decorative elements, or shadows unless explicitly required. The final "
    "result must look like a clean marketplace main product image, not a lifestyle photo."
)

_TAIL = "1:1 aspect ratio, 2048x2048 pixels resolution, clean e-commerce cutout style."


def _sanitize(text: str) -> str:
    text = re.sub(r"```\w*", "", text).replace("**", "").strip().strip('"“”\'')
    text = " ".join(text.split())
    return text if text.endswith(_TAIL) else f"{text.rstrip('.')}. {_TAIL}"


async def style_references(db, store_id: int, exclude_product_id: int | None = None, limit: int = 2):
    """Main photos of the shop's earlier products, so a new photo matches the catalogue's look."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.models import Product

    q = (select(Product).where(Product.store_id == store_id, Product.id != (exclude_product_id or 0))
         .options(selectinload(Product.images)).order_by(Product.is_golden_sample.desc(), Product.id.desc()).limit(30))
    refs = []
    for prod in (await db.execute(q)).scalars():
        for img in sorted(prod.images, key=lambda i: i.position)[:1]:
            if loaded := read_local(img.url):
                refs.append(loaded)
        if len(refs) >= limit:
            break
    return refs


DESIGN_IMAGE_SYSTEM = (
    "You are generating a clean marketplace main product image. The first attached image is a DESIGN (artwork, logo, "
    "pattern or print), not a photo of the product: show the product described in the prompt with that design applied "
    "to it exactly, keeping the design's shapes, proportions and line work. Never add props, text overlays, watermarks, "
    "people or extra objects unless the prompt requires them."
)

PHOTO_KIND_SYSTEM = """Look at the attached image from an online seller. Decide what it is.
"product": a photo of a real, physical product (even a raw, badly lit phone photo, on a person, a hanger or a table).
"design": a flat artwork, logo, drawing, pattern or print file the seller wants printed on a product (no physical product visible).
Respond ONLY with JSON: {"kind":"product"} or {"kind":"design"}"""


def _builder_input(*, title: str, note: str, category: str, has_photo: bool, styles: int, design: bool = False) -> str:
    num, label = IMAGE_CATEGORIES.get(category, IMAGE_CATEGORIES["general"])
    lines = [f"PRODUCT NAME: {title}", f"user description: {note or title}",
             f"Selected category (prompt_type {num}): {label}", ""]
    if has_photo and design:
        lines.append("Attached image 1 is NOT the product: it is the DESIGN (artwork/print) the seller puts on the product. "
                     "Write a text-to-image prompt that depicts the product exactly as the name and description say, with "
                     "this design applied to it (for clothing: printed on the chest, centred, at a natural print size, in a "
                     "colour that contrasts with the garment unless the description says otherwise). The design must be "
                     "reproduced exactly: same shapes, proportions and line work. Do not invent other logos or text.")
    elif has_photo:
        lines.append("Attached image 1 is the product photo (the ATTACHED IMAGE).")
    else:
        lines.append("There is NO product photo: write a text-to-image prompt that depicts the product exactly as the "
                     "name and description say. Do not invent logos, brand names, printed text or details not stated.")
    if styles:
        first = 2 if has_photo else 1
        lines.append(f"Attached images {first}-{first + styles - 1} are STYLE REFERENCES: photos of this shop's other products. "
                     "The new photo must look like it belongs to the same catalogue: match their background, lighting, "
                     "camera angle, framing and layout. These style references override the default pure-white background "
                     "rule when they use a different background. Never copy the products shown in them.")
    lines += ["", "Analyze the attached e-commerce image and return exactly one raw English paragraph."
              if has_photo and not design else "Return exactly one raw English paragraph."]
    return "\n".join(lines)


async def chosen_styles(db, store_id: int, urls: list[str], product_ids: list[int], limit: int = 3):
    """Style samples the seller picked: uploaded photos and/or the main photo of chosen products."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.models import Product

    picked = [u for u in urls if u.startswith(f"/media/{store_id}/")]
    if product_ids:
        q = (select(Product).where(Product.store_id == store_id, Product.id.in_(product_ids))
             .options(selectinload(Product.images)))
        for prod in (await db.execute(q)).scalars():
            picked += [i.url for i in sorted(prod.images, key=lambda i: i.position)[:1]]
    return [b for b in (read_local(u) for u in picked) if b][:limit]


async def make_photo(ctx, *, store_id: int, title: str, note: str, category: str = "general",
                     image_url: str | None = None, product_id: int | None = None,
                     style_urls: list[str] | None = None, style_product_ids: list[int] | None = None,
                     design: bool = False) -> str:
    """Cleans `image_url`, or generates a photo from the text when there is none, in the style of the samples the
    seller picked, else of the shop's earlier product photos. `design`: the photo is artwork to put on the product. Returns the /media URL. Raises AIError/ValueError with a Persian message."""
    loaded = read_local(image_url) if image_url else None
    if image_url and not loaded:
        raise ValueError("عکس اصلی روی سرور پیدا نشد.")
    styles = []
    if ctx.db is not None:
        styles = await chosen_styles(ctx.db, store_id, style_urls or [], style_product_ids or [])
        styles = styles or await style_references(ctx.db, store_id, product_id)
    images = ([loaded] if loaded else []) + styles
    design = design and bool(loaded)
    async with ctx.step("image_prompt", "تحلیل عکس و نوشتن دستور ویرایش" if loaded and not design else "نوشتن دستور ساخت عکس",
                        request={"category": category, "has_photo": bool(loaded), "design": design,
                                 "style_refs": len(styles)}) as s:
        raw = await ai.generate_text(
            task="vision", system=PROMPT_BUILDER_SYSTEM, images=images, store_id=store_id,
            prompt=_builder_input(title=title, note=note, category=category, has_photo=bool(loaded), styles=len(styles),
                                  design=design))
        prompt = _sanitize(raw)
        s.response = {"prompt": prompt}
    label = "تمیز کردن عکس" if loaded and not design else "ساخت عکس محصول"
    async with ctx.step("image_edit", f"{label} (به سبک عکس‌های قبلی فروشگاه)" if styles else label) as s:
        system = DESIGN_IMAGE_SYSTEM if design else IMAGE_SYSTEM if loaded else (
            "You are generating a clean marketplace main product image from a text prompt. Depict only the described "
            "product, accurately; never add props, text overlays, watermarks, people or extra objects unless the prompt "
            "requires them.")
        if styles:
            system += (" The last attached images are style references from the same shop: match their background, "
                       "lighting, angle and framing, but never copy their products.")
        data, mime = await ai.generate_image(system=system, prompt=prompt, images=images, store_id=store_id)
        url = save_bytes(store_id, data, content_type=mime)
        s.response = {"url": url, "bytes": len(data), "mime": mime}
    return url


async def photo_kind(store_id: int, url: str) -> str:
    """Cheap first guess for the form: is this a product photo or a design to print? Falls back to 'product'."""
    loaded = read_local(url)
    if not loaded or not url.startswith(f"/media/{store_id}/"):
        return "product"
    try:
        res = await ai.generate_json(task="vision", system=PHOTO_KIND_SYSTEM, prompt="Classify this image.",
                                     images=[loaded], store_id=store_id)
    except ai.AIError:
        return "product"
    return "design" if res.get("kind") == "design" else "product"
