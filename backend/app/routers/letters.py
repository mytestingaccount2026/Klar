"""Letter upload + retrieval endpoints (auth-required, /api/letters/*)."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.auth.dependencies import get_current_user
from app.database import get_session
from app.errors import ErrorCode, KlarHTTPException
from app.models import (
    ActionItem,
    DocumentCategory,
    Letter,
    LetterStatus,
    User,
    utcnow,
)
from app.schemas import (
    ErrorResponse,
    LetterListItem,
    LetterResponse,
    LetterUploadResponse,
)
from app.services.extraction import extract_from_letter_file, normalize_lang
from app.services.persistence import persist_extraction
from app.services.storage import detect_magic_mime, save_letter_file

router = APIRouter(prefix="/api/letters", tags=["letters"])

ACCEPTED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/webp",
    "application/pdf",
}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per spec


# --- helpers --------------------------------------------------------------


def _letter_response(letter: Letter, actions: list[ActionItem]) -> LetterResponse:
    return LetterResponse(
        id=letter.id,
        institution=letter.institution,
        document_type=letter.document_type,
        letter_type=letter.letter_type or letter.document_type,
        category=letter.category,
        summary=letter.summary,
        language=letter.language,
        risk_score=letter.risk_score,
        deadline_date=letter.deadline_date,
        explanation=letter.explanation,
        response_draft=letter.response_draft,
        checklist=letter.checklist,
        citations=letter.citations,
        consequence=letter.consequence,
        status=letter.status,
        processed_at=letter.processed_at,
        created_at=letter.created_at,
        actions=[
            {
                "id": str(a.id),
                "title": a.title,
                "description": a.description,
                "deadline": a.deadline.isoformat() if a.deadline else None,
                "deadline_source": a.deadline_source.value,
                "deadline_confidence": a.deadline_confidence,
                "severity": a.severity.value,
                "status": a.status.value,
                "steps": a.steps,
                "reply_needed": a.reply_needed,
                "evidence_span": a.evidence_span,
            }
            for a in actions
        ],
        extraction_warnings=letter.extraction_warnings,
    )


# --- POST /api/letters/upload --------------------------------------------


@router.post(
    "/upload",
    response_model=LetterUploadResponse,
    status_code=201,
    summary="Upload a Brief (image or PDF)",
    description=(
        "Multipart upload of a German letter. Returns immediately with the new "
        "`letter_id`; the heavy AI extraction runs LATER via "
        "`GET /api/letters/{letter_id}/process` (SSE).\n\n"
        "Accepts: JPEG / PNG / WebP / HEIC / PDF, up to 10 MB. Server verifies "
        "magic bytes against the declared `Content-Type` — defends against "
        "binaries renamed `.jpg`."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "`LETTER_EMPTY_UPLOAD`."},
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        413: {"model": ErrorResponse, "description": "`LETTER_TOO_LARGE` (>10 MB)."},
        415: {
            "model": ErrorResponse,
            "description": (
                "`LETTER_UNSUPPORTED_TYPE` (bad MIME), "
                "`LETTER_CORRUPT_FILE` (no magic match), or "
                "`LETTER_MIME_MISMATCH` (declared ≠ detected)."
            ),
        },
    },
)
async def upload_letter(
    file: UploadFile = File(...),
    lang: str | None = Query(default=None, max_length=8),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Save the upload, create a Letter row with status='uploaded', return its id.

    Extraction runs LATER via GET /api/letters/{id}/process (SSE).

    Validation:
    - Declared Content-Type must be in ACCEPTED_MIMES.
    - File MUST match its declared type via magic-bytes inspection — defends
      against malicious renames (e.g. an executable announced as image/jpeg).
    - 10 MB max size per spec.
    """
    if not file.content_type or file.content_type not in ACCEPTED_MIMES:
        raise KlarHTTPException(415, ErrorCode.LETTER_UNSUPPORTED_TYPE)

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise KlarHTTPException(400, ErrorCode.LETTER_EMPTY_UPLOAD)
    if len(image_bytes) > MAX_FILE_BYTES:
        raise KlarHTTPException(413, ErrorCode.LETTER_TOO_LARGE)

    actual_mime = detect_magic_mime(image_bytes)
    if actual_mime is None:
        raise KlarHTTPException(415, ErrorCode.LETTER_CORRUPT_FILE)
    # Allow image/jpeg ↔ image/jpg variants; otherwise demand strict match.
    if actual_mime != file.content_type and not (
        actual_mime.startswith("image/")
        and file.content_type.startswith("image/")
        and actual_mime.split("/")[-1] == file.content_type.split("/")[-1]
    ):
        raise KlarHTTPException(
            415,
            ErrorCode.LETTER_MIME_MISMATCH,
            details={"declared": file.content_type, "detected": actual_mime},
        )

    out_lang = normalize_lang(lang or user.language)

    letter = Letter(
        user_id=user.id,
        language=out_lang,
        status=LetterStatus.UPLOADED,
    )
    db.add(letter)
    db.flush()  # need letter.id for filename

    saved_path = save_letter_file(user.id, letter.id, actual_mime, image_bytes)
    letter.original_file = saved_path
    db.add(letter)
    db.commit()
    db.refresh(letter)

    return LetterUploadResponse(letter_id=letter.id)


# --- POST /api/letters/{id}/extract (non-streaming convenience) ----------


@router.post(
    "/{letter_id}/extract",
    response_model=LetterResponse,
    summary="Synchronous extraction (no streaming)",
    description=(
        "Runs ONE structured Qwen call and returns the full LetterResponse. "
        "Long-form fields (`explanation`, `response_draft`, `checklist`, "
        "`citations`) are NOT populated here — use the SSE `/process` route "
        "for those. Used by clients that don't support EventSource."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
        409: {"model": ErrorResponse, "description": "`LETTER_FILE_MISSING`."},
        502: {"model": ErrorResponse, "description": "`EXTRACTION_FAILED`."},
    },
)
async def extract_letter(
    letter_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Synchronous extraction — runs the structured Qwen call inline and
    returns the populated LetterResponse. Used by clients that don't want SSE.

    Long-form fields (explanation, response_draft, checklist, citations) are
    NOT populated here — only the SSE endpoint generates those.
    """
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    if not letter.original_file:
        raise KlarHTTPException(409, ErrorCode.LETTER_FILE_MISSING)

    letter.status = LetterStatus.PROCESSING
    db.add(letter)
    db.commit()

    try:
        mime = (
            "application/pdf"
            if letter.original_file.lower().endswith(".pdf")
            else "image/jpeg"
        )
        extracted = await extract_from_letter_file(
            letter.original_file, mime, lang=letter.language
        )
    except Exception:
        letter.status = LetterStatus.ERROR
        db.add(letter)
        db.commit()
        # Don't leak the raw exception. Logged by unhandled_exception_handler.
        raise KlarHTTPException(502, ErrorCode.EXTRACTION_FAILED)

    actions = persist_extraction(db, letter, extracted)
    letter.status = LetterStatus.COMPLETED
    letter.processed_at = utcnow()
    db.add(letter)
    db.commit()
    db.refresh(letter)

    return _letter_response(letter, actions)


# --- GET /api/letters ----------------------------------------------------


@router.get(
    "",
    response_model=list[LetterListItem],
    summary="List the current user's letters",
    description=(
        "Sorted by `created_at` DESC. Optional filters:\n"
        "- `?status=uploaded|processing|completed|error`\n"
        "- `?category=health_insurance|tax|immigration|...` (see DocumentCategory enum)\n\n"
        "Empty-string values for `status` / `category` are treated as 'no filter' "
        "— safe to bind directly to React `useState('')`."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        422: {
            "model": ErrorResponse,
            "description": "Unknown status or category value.",
        },
    },
)
def list_letters(
    # `str | None` (not `LetterStatus | None`) so empty-string query params
    # (?status=) don't trip Pydantic enum coercion. We parse manually below.
    status: str | None = Query(default=None),
    category: str | None = Query(default=None),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    parsed_status: LetterStatus | None = None
    if status:
        try:
            parsed_status = LetterStatus(status)
        except ValueError:
            raise KlarHTTPException(
                422,
                ErrorCode.VALIDATION_ERROR,
                message=f"Unknown status: {status!r}.",
                details={
                    "errors": [
                        {
                            "field": "status",
                            "message": "must be one of "
                            + ", ".join(s.value for s in LetterStatus),
                        }
                    ]
                },
            )
    parsed_category: DocumentCategory | None = None
    if category:
        try:
            parsed_category = DocumentCategory(category)
        except ValueError:
            raise KlarHTTPException(
                422,
                ErrorCode.VALIDATION_ERROR,
                message=f"Unknown category: {category!r}.",
                details={
                    "errors": [
                        {
                            "field": "category",
                            "message": "must be one of "
                            + ", ".join(c.value for c in DocumentCategory),
                        }
                    ]
                },
            )

    stmt = select(Letter).where(Letter.user_id == user.id)
    if parsed_status:
        stmt = stmt.where(Letter.status == parsed_status)
    if parsed_category:
        stmt = stmt.where(Letter.category == parsed_category)
    stmt = stmt.order_by(Letter.created_at.desc())
    items = list(db.scalars(stmt).all())
    return [
        LetterListItem(
            id=letter.id,
            letter_type=letter.letter_type or letter.document_type,
            category=letter.category,
            risk_score=letter.risk_score,
            deadline_date=letter.deadline_date,
            status=letter.status,
            created_at=letter.created_at,
        )
        for letter in items
    ]


# --- GET /api/letters/{id}/process (SSE) ---------------------------------


@router.get(
    "/{letter_id}/process",
    summary="SSE stream: full extraction + long-form generation",
    description=(
        "Server-Sent Events stream that orchestrates the AI pipeline. The "
        "response is `text/event-stream`; each frame has `event: <type>` and "
        "`data: <json>` lines.\n\n"
        "Event sequence (each event JSON shape is documented in schemas.py):\n"
        "1. `ocr_result` — verbatim German text (one frame)\n"
        "2. `classification` — document type + category + agency\n"
        "3. `risk_score` — integer 0–100 + label\n"
        "4. `deadline` — earliest action's deadline + days remaining\n"
        "5. `consequence` — what happens if the user does nothing\n"
        "6. `explanation` — streaming, multiple frames, one chunk per frame\n"
        "7. `response_draft` — streaming, conditional (only if any action "
        "needs a reply); ALWAYS in German regardless of `?lang`\n"
        "8. `checklist` — array of items the user must gather\n"
        "9. `citations` — RAG corpus hits used to ground the extraction\n"
        "10. `done` — final frame, `{ letter_id }`. Frontend should close the EventSource.\n"
        "\nOn any failure, a single `error` frame is sent with the standard "
        "Klar error envelope (`code` + `message`), then the stream closes.\n\n"
        "**Frontend:** open with `new EventSource(url, { withCredentials: true })` "
        "so the `klar_session` cookie travels."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream — see description for event sequence.",
        },
    },
)
async def process_letter(
    letter_id: UUID,
    lang: str | None = Query(default=None, max_length=8),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """SSE stream: orchestrates extraction + long-form generation.

    Auth is via the session cookie (set by /api/auth/login). EventSource
    callers must construct with `withCredentials: true` so the cookie travels
    on the SSE GET. The spec's `?token=` query-param is intentionally NOT
    supported — tokens-in-URL get logged to access logs and browser history.
    See docs/02-backend.md "Modifications" section.

    NOTE: we do NOT pass the injected `db` to the generator — FastAPI closes
    it as soon as this handler returns. The generator opens and owns its own
    Session for the stream's lifetime.
    """
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    out_lang = normalize_lang(lang or letter.language or user.language)

    # Import here, not at module top, so pipeline can freely import from
    # routers (no cycle, since orchestrator only depends on services/).
    from app.pipeline.orchestrator import process_letter_stream

    return StreamingResponse(
        process_letter_stream(letter_id, out_lang),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --- GET /api/letters/{id} ------------------------------------------------


@router.get(
    "/{letter_id}",
    response_model=LetterResponse,
    summary="Get a single letter with all its actions",
    description=(
        "Returns the full `LetterResponse` including denormalized "
        "`risk_score`, `deadline_date`, every `ActionItem` (with "
        "`evidence_span`, `deadline_confidence`, `steps`), plus any long-form "
        "fields (`explanation`, `response_draft`, `checklist`, `citations`) "
        "the `/process` pipeline has already filled in."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
    },
)
def get_letter(
    letter_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)
    actions = list(
        db.scalars(select(ActionItem).where(ActionItem.letter_id == letter_id)).all()
    )
    return _letter_response(letter, actions)
