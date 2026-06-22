"""Service layer: pure functions called by routers and the pipeline."""

from app.services.extraction import (
    DOCUMENT_CATEGORIES,
    SUPPORTED_LANGS,
    extract_from_letter_file,
    generate_checklist,
    normalize_lang,
    stream_explanation,
    stream_response_draft,
)
from app.services.pdf_pages import (
    iter_data_urls,
    pdf_to_image_bytes,
    split_to_image_bytes,
)
from app.services.persistence import persist_extraction
from app.services.risk import compute_risk
from app.services.storage import detect_magic_mime, is_pdf, save_letter_file, user_dir

__all__ = [
    "DOCUMENT_CATEGORIES",
    "SUPPORTED_LANGS",
    "extract_from_letter_file",
    "generate_checklist",
    "normalize_lang",
    "stream_explanation",
    "stream_response_draft",
    "iter_data_urls",
    "pdf_to_image_bytes",
    "split_to_image_bytes",
    "persist_extraction",
    "compute_risk",
    "detect_magic_mime",
    "is_pdf",
    "save_letter_file",
    "user_dir",
]
