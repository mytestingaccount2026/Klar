"""Convert a PDF file into a list of PNG image bytes, one per page.

Requires `poppler` to be installed on the host:
    macOS:   brew install poppler
    Linux:   apt-get install poppler-utils
"""

from io import BytesIO
from typing import Iterable


def pdf_to_image_bytes(
    path: str, *, dpi: int = 200, max_pages: int = 12
) -> list[bytes]:
    """Render up to `max_pages` pages of `path` to PNG bytes.

    Imported lazily so callers that never touch a PDF don't pay the
    pdf2image / poppler import cost.
    """
    from pdf2image import convert_from_path

    pages = convert_from_path(path, dpi=dpi, first_page=1, last_page=max_pages)
    out: list[bytes] = []
    for img in pages:
        buf = BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def split_to_image_bytes(path: str, mime: str) -> list[tuple[bytes, str]]:
    """Return a list of (image_bytes, mime) for any supported letter file.

    Single image files return a one-element list. PDFs return one element per
    rendered page (PNG).
    """
    if mime == "application/pdf" or path.lower().endswith(".pdf"):
        return [(b, "image/png") for b in pdf_to_image_bytes(path)]

    with open(path, "rb") as f:
        return [(f.read(), mime or "image/jpeg")]


def iter_data_urls(pages: Iterable[tuple[bytes, str]]) -> Iterable[dict]:
    """Yield OpenAI-compatible image content parts from (bytes, mime) tuples."""
    import base64

    for data, mime in pages:
        b64 = base64.b64encode(data).decode()
        yield {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
