"""Raw seller photo -> clean marketplace main image, ported from platform-wrappers (digikala.api image pipeline).

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


async def clean_photo(ctx, *, store_id: int, image_url: str, title: str, note: str, category: str = "general") -> str:
    """Returns the /media URL of the cleaned image. Raises AIError/ValueError with a Persian message on failure."""
    loaded = read_local(image_url)
    if not loaded:
        raise ValueError("عکس اصلی روی سرور پیدا نشد.")
    num, label = IMAGE_CATEGORIES.get(category, IMAGE_CATEGORIES["general"])
    async with ctx.step("image_prompt", "تحلیل عکس و نوشتن دستور ویرایش", request={"category": label}) as s:
        raw = await ai.generate_text(
            task="vision", system=PROMPT_BUILDER_SYSTEM, images=[loaded], store_id=store_id,
            prompt=(f"PRODUCT NAME: {title}\nuser description: {note or title}\n"
                    f"Selected category (prompt_type {num}): {label}\n\n"
                    "Analyze the attached e-commerce image and return exactly one raw English paragraph."))
        prompt = _sanitize(raw)
        s.response = {"prompt": prompt}
    async with ctx.step("image_edit", "تمیز کردن عکس (پس‌زمینهٔ سفید، وسط کادر)") as s:
        data, mime = await ai.generate_image(system=IMAGE_SYSTEM, prompt=prompt, images=[loaded], store_id=store_id)
        url = save_bytes(store_id, data, content_type=mime)
        s.response = {"url": url, "bytes": len(data), "mime": mime}
    return url
