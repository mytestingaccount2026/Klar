"""
Form-fill: Qwen-VL detects blank fields with bbox_2d coordinates,
then Pillow draws red placeholder text at exact positions.

Qwen-VL returns bbox_2d as [x1, y1, x2, y2] in 0-1000 normalized range.
We map to actual pixels: pixel = coord / 1000 * dimension.
"""

import base64
import json
import os
import re
from io import BytesIO

import httpx
from PIL import Image, ImageDraw, ImageFont

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.environ.get(
    "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

DETECT_PROMPT = """Detect all EMPTY blank lines, empty boxes, and unfilled form fields in this German document image where a person needs to handwrite their information.

For each blank field found, return its location using bbox_2d format and what should be written there.

Return a JSON array. Each item:
{"bbox_2d": [x1, y1, x2, y2], "label": "field name in German", "placeholder": "what to write in English"}

Rules:
- bbox_2d coordinates are [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
- ONLY detect genuinely EMPTY/BLANK fields — skip anything with printed text already in it
- placeholder must be in English describing what the user fills in
- Return ONLY the JSON array, no other text"""


async def _detect_fields(image_path: str) -> list[dict]:
    """Call Qwen-VL to detect blank form fields with bbox_2d coordinates."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode()
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen-vl-max",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                            {"type": "text", "text": DETECT_PROMPT},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 2048,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    # Parse JSON from model output
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            s = part.strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
            if s.startswith("["):
                text = s
                break

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    json_str = text[start : end + 1]
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        fields = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    return fields if isinstance(fields, list) else []


def _draw_placeholders(image_path: str, fields: list[dict]) -> bytes:
    """Draw red placeholder text on the original image using Pillow.

    bbox_2d from Qwen-VL is in 0-1000 normalized range.
    Convert: pixel = coord / 1000 * image_dimension
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    font_size = max(14, img.height // 50)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    red = (220, 38, 38)

    # Compute scale from model's coordinate space to actual pixels.
    # Qwen-VL returns coords in its internal resolution, not 0-1000.
    # Derive scale from the max coordinate values in the response.
    all_bboxes = [
        f["bbox_2d"]
        for f in fields
        if isinstance(f.get("bbox_2d"), list) and len(f["bbox_2d"]) == 4
    ]
    if not all_bboxes:
        buf = BytesIO()
        img.save(buf, format="PNG", quality=95)
        return buf.getvalue()

    coord_max_x = max(b[2] for b in all_bboxes)
    coord_max_y = max(b[3] for b in all_bboxes)
    # The max coord is near but not at the image edge — add small padding
    scale_x = img_w / (coord_max_x * 1.02)
    scale_y = img_h / (coord_max_y * 1.02)

    for field in fields:
        try:
            bbox = field.get("bbox_2d")
            if not bbox or len(bbox) != 4:
                continue

            # Scale from model coords to actual pixels
            x1 = int(float(bbox[0]) * scale_x)
            y1 = int(float(bbox[1]) * scale_y)
            x2 = int(float(bbox[2]) * scale_x)
            y2 = int(float(bbox[3]) * scale_y)

            # Clamp
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1 + 20, min(x2, img_w))
            y2 = max(y1 + font_size + 6, min(y2, img_h))

            placeholder = str(field.get("placeholder", "FILL IN"))

            # Measure how wide the placeholder text actually is
            text_bbox = font.getbbox(placeholder)
            text_w = text_bbox[2] - text_bbox[0] + 12  # small padding
            text_h = text_bbox[3] - text_bbox[1] + 6

            # Shrink the box to fit just the text, not the full detected width
            box_w = min(text_w, x2 - x1)
            box_h = max(text_h, y2 - y1)

            # Draw a tight highlight behind the text only
            draw.rectangle(
                [x1, y1, x1 + box_w, y1 + box_h],
                fill=(255, 235, 235),
                outline=red,
                width=1,
            )

            # Draw text
            text_y = y1 + max(0, (box_h - font_size) // 2)
            draw.text((x1 + 6, text_y), placeholder, fill=red, font=font)

        except (KeyError, ValueError, TypeError):
            continue

    buf = BytesIO()
    img.save(buf, format="PNG", quality=95)
    return buf.getvalue()


async def generate_filled_form(image_path: str, placeholders: list[str]) -> bytes:
    """Detect blank fields with Qwen-VL, draw placeholders with Pillow."""
    fields = await _detect_fields(image_path)
    if not fields:
        with open(image_path, "rb") as f:
            return f.read()
    return _draw_placeholders(image_path, fields)
