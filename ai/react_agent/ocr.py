import base64
import os

import httpx

from ai.prompts import OCR_PROMPT

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.environ.get(
    "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Dedicated OCR model — faster and more accurate for text extraction
QWEN_OCR_MODEL = os.environ.get("QWEN_OCR_MODEL", "qwen-vl-ocr")


async def extract_text_from_image(image_path: str) -> str:
    """Send image to Qwen-VL-OCR and return extracted text."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Detect MIME type
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    mime_type = mime_map.get(ext, "image/jpeg")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_OCR_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "min_pixels": 3072,
                                    "max_pixels": 8388608,
                                },
                            },
                            {"type": "text", "text": OCR_PROMPT},
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
