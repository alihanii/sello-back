"""Raw seller photo (or just the product text) -> marketplace main image, ported from platform-wrappers (digikala.api image pipeline).

Two calls, like the original: a text model that can see images writes a precise English img2img prompt from the
photo (prompt builder), then a Gemini image model redraws the same photo with that prompt: same product, pure
white background, centred, square 2K. Prompts are copied verbatim from platform-wrappers (V0.2.0)."""

import io
import re

from PIL import Image, ImageDraw

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

PEOPLE_SYSTEM = """Look at the attached product photo from an online seller.
Find the product being sold, and any real person in the photo.
Boxes are [ymin, xmin, ymax, xmax] on a 0-1000 scale.
"product_box": the tightest box around the product only (for worn clothing: the garment, from its top edge, e.g. the
collar or hood, to its bottom edge). "faces": a box around each visible human head and face (with hair and sunglasses).
Respond ONLY with JSON: {"people": true|false, "product_box": [..], "faces": [[..], ..]}"""

_TAIL = "1:1 aspect ratio, 2048x2048 pixels resolution, clean e-commerce cutout style."


def _sanitize(text: str) -> str:
    """platform-wrappers sanitize_generated_prompt, verbatim."""
    cleaned = text.replace("**", "").strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip().strip('"')


def _builder_input(*, title: str, note: str, category: str, has_photo: bool, design: bool = False,
                   no_people: bool = False) -> str:
    num, label = IMAGE_CATEGORIES.get(category, IMAGE_CATEGORIES["general"])
    if has_photo and not design and not no_people:
        # platform-wrappers build_prompt_builder_user_input, verbatim: the seller's text is the instruction.
        instruction = note or title
        return (f"PRODUCT NAME: {instruction}\n"
                f"user description: {instruction}\n"
                f"Selected category (prompt_type {num}): {label}\n\n"
                "Analyze the attached e-commerce image and return exactly one raw English paragraph.")
    lines = [f"PRODUCT NAME: {title}", f"user description: {note or title}",
             f"Selected category (prompt_type {num}): {label}", ""]
    if no_people:
        lines += ["Attached image 1 is the product photo (the ATTACHED IMAGE).",
                  "The product was worn or held by a person whose face has been blanked. Show the product ALONE: no "
                  "person, no body, no skin, no hands, no face (clothing: invisible ghost mannequin or flat lay). Keep "
                  "the product's own prints and text. Do not mention the person or any identity in the prompt.",
                  "", "Analyze the attached e-commerce image and return exactly one raw English paragraph."]
        return "\n".join(lines)
    if has_photo and design:
        lines.append("Attached image 1 is NOT the product: it is the DESIGN (artwork/print) the seller puts on the product. "
                     "Write a text-to-image prompt that depicts the product exactly as the name and description say, with "
                     "this design applied to it (for clothing: printed on the chest, centred, at a natural print size, in a "
                     "colour that contrasts with the garment unless the description says otherwise). The design must be "
                     "reproduced exactly: same shapes, proportions and line work. Do not invent other logos or text.")
    else:
        lines.append("There is NO product photo: write a text-to-image prompt that depicts the product exactly as the "
                     "name and description say. Do not invent logos, brand names, printed text or details not stated.")
    lines += ["", "Return exactly one raw English paragraph."]
    return "\n".join(lines)


async def make_photo(ctx, *, store_id: int, title: str, note: str, category: str = "general",
                     image_url: str | None = None, design: bool = False) -> str:
    """Cleans the seller's photo the way platform-wrappers made Digikala main images, or, when the photo is a design
    (`design`) or missing, generates the product photo from the description (and the design). Only the seller's own
    photo is sent. Returns the /media URL. Raises AIError/ValueError with a Persian message."""
    loaded = read_local(image_url) if image_url else None
    if image_url and not loaded:
        raise ValueError("عکس اصلی روی سرور پیدا نشد.")
    design = design and bool(loaded)
    no_people = False
    if loaded and not design and category == "modeled_clothing":
        # platform-wrappers asks for a new model here, and image models refuse to swap a real person (IMAGE_OTHER).
        # So for worn clothing the face goes first and the garment is shown alone.
        if hidden := await _hide_people(ctx, store_id, loaded):
            loaded, no_people, category = hidden, True, "flat_clothing"
    images = [loaded] if loaded else []
    async with ctx.step("image_prompt", "تحلیل عکس و نوشتن دستور ویرایش" if loaded and not design else "نوشتن دستور ساخت عکس",
                        request={"category": category, "has_photo": bool(loaded), "design": design}) as s:
        raw = await ai.generate_text(
            task="image_prompt" if await ai.has_route("image_prompt") else "vision",
            system=PROMPT_BUILDER_SYSTEM, images=images, store_id=store_id,
            prompt=_builder_input(title=title, note=note, category=category, has_photo=bool(loaded), design=design,
                                  no_people=no_people))
        prompt = _sanitize(raw)
        s.response = {"prompt": prompt}
    label = "تمیز کردن عکس" if loaded and not design else "ساخت عکس محصول"
    system = DESIGN_IMAGE_SYSTEM if design else IMAGE_SYSTEM if loaded else (
        "You are generating a clean marketplace main product image from a text prompt. Depict only the described "
        "product, accurately; never add props, text overlays, watermarks, people or extra objects unless the prompt "
        "requires them.")
    try:
        async with ctx.step("image_edit", label, request={"prompt": prompt, "images": len(images), "design": design}) as s:
            data, mime = await ai.generate_image(system=system, prompt=prompt, images=images, store_id=store_id)
            s.response = {"bytes": len(data), "mime": mime}
    except ai.AIError as exc:
        if exc.code != "ai_no_image":
            raise
        # The model returns no picture (IMAGE_OTHER) for a real person's face or a long, conflicting prompt.
        # One retry with a short, plain instruction, on the photo without its faces.
        if loaded and not design and not no_people and (hidden := await _hide_people(ctx, store_id, loaded)):
            images, no_people = [hidden], True
        simple = _simple_prompt(title=title, note=note, has_photo=bool(loaded), design=design, no_people=no_people)
        async with ctx.step("image_retry", "تلاش دوباره با دستور ساده‌تر",
                            request={"prompt": simple, "images": 1 if loaded else 0, "design": design}) as s:
            data, mime = await ai.generate_image(system=system, prompt=simple, images=images, store_id=store_id)
            s.response = {"bytes": len(data), "mime": mime}
    url = save_bytes(store_id, data, content_type=mime)
    return url


def _simple_prompt(*, title: str, note: str, has_photo: bool, design: bool, no_people: bool = False) -> str:
    what = f"{title}. {note}".strip(". ")
    if has_photo and design:
        head = f"Product photo of: {what}. Apply the attached design to the product exactly as drawn."
    elif has_photo:
        head = ("Clean product photo of the product in the attached image, keeping its colours, prints and details."
                + (" Show it alone, as an invisible ghost mannequin or flat lay; remove the person." if no_people else ""))
    else:
        head = f"Clean product photo of: {what}."
    return f"{head} Centred, fully visible, plain white background, soft even light, no added text or props. {_TAIL}"


async def _hide_people(ctx, store_id: int, loaded: tuple[bytes, str]) -> tuple[bytes, str] | None:
    """Crops the photo to the product and blanks every face. None when there is no person (or the check fails)."""
    try:
        async with ctx.step("image_people", "حذف آدم از عکس", request={"images": 1}) as s:
            res = await ai.generate_json(task="vision", system=PEOPLE_SYSTEM, prompt="Find the product and the people.",
                                         images=[loaded], store_id=store_id)
            s.response = res
    except ai.AIError:
        return None
    if not res.get("people"):
        return None
    return crop_without_faces(loaded[0], res.get("product_box"), res.get("faces") or [])


def _box(raw, w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        y0, x0, y1, x1 = (min(max(float(v), 0), 1000) for v in raw)
    except (TypeError, ValueError):
        return None
    box = (round(x0 * w / 1000), round(y0 * h / 1000), round(x1 * w / 1000), round(y1 * h / 1000))
    return box if box[2] > box[0] and box[3] > box[1] else None


def crop_without_faces(data: bytes, product_box, faces: list) -> tuple[bytes, str] | None:
    """Paints the faces white, then crops to the product (with a small margin). Returns PNG bytes."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = img.size
    draw = ImageDraw.Draw(img)
    face_boxes = [b for b in (_box(f, w, h) for f in faces if isinstance(f, list | tuple)) if b]
    for x0, y0, x1, y1 in face_boxes:
        pad = round(0.1 * max(x1 - x0, y1 - y0))
        draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill="white")
    crop = _box(product_box, w, h) if isinstance(product_box, list | tuple) else None
    if crop:
        m = round(0.03 * max(w, h))
        x0, y0, x1, y1 = max(crop[0] - m, 0), max(crop[1] - m, 0), min(crop[2] + m, w), min(crop[3] + m, h)
        img = img.crop((x0, y0, x1, y1))
    elif not face_boxes:
        return None
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue(), "image/png"


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
