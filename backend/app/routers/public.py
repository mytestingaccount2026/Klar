"""Frontend-facing adapter routes — root-level (no /api prefix).

The frontend team's contract (`docs/06-frontend-integration-contract.md`)
expects:

    POST   /letters?lang=        — synchronous upload + extract, returns full Letter
    GET    /letters/{id}?lang=   — returns full Letter
    GET    /actions?lang=&status= — returns ActionListItem[]
    PATCH  /actions/{id}         — returns {id, status}
    POST   /rag/search           — returns {hits: RagHit[]}
    GET    /health               — already at root, no change

This module mounts those five routes at root. Each route is auth-required via
the session cookie (frontend bootstraps once via /auth/signup; see docs/
06-api-contract.md → "Frontend auth bootstrap"). All routes delegate to the
existing services in app/services/* — no duplicated extraction or persistence
logic.

The `/api/*` routes (auth, /api/letters, /api/actions, /api/deadlines,
/api/rag) are unchanged and continue to serve clients that want the richer
shapes / SSE streaming.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.auth.dependencies import get_current_user
from app.database import get_session
from app.errors import ErrorCode, KlarHTTPException
from app.models import (
    ActionItem,
    ActionStatus,
    Letter,
    LetterStatus,
    RiskScore,
    User,
    utcnow,
)
from app.schemas import (
    ActionUpdate,
    CitationItem,
    ErrorResponse,
    LetterUploadResponse,
    PublicAction,
    PublicActionListItem,
    PublicActionUpdateResponse,
    PublicLetter,
    ChatRequest,
    ChatResponse,
    RagQuery,
    RagResponse,
    ReplyDraft,
    ReplyRequest,
    RiskBreakdown,
)
from app.services import ai_bridge
from app.services.extraction import (
    extract_from_letter_file,
    normalize_lang,
)
from app.services.persistence import persist_extraction
from app.services.storage import detect_magic_mime, save_letter_file

logger = logging.getLogger("klar.public")

router = APIRouter(tags=["public"])

ACCEPTED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/webp",
    "application/pdf",
}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per spec


# ---------- shape projection: Letter + ActionItems → PublicLetter ----------


def _public_action(action: ActionItem, latest_risk: RiskScore | None) -> PublicAction:
    risk_breakdown = None
    if latest_risk is not None:
        risk_breakdown = RiskBreakdown(
            score=latest_risk.score,
            deadline_proximity_pts=latest_risk.deadline_proximity_pts,
            institution_weight=latest_risk.institution_weight,
            severity_pts=latest_risk.severity_pts,
            missing_info_penalty=latest_risk.missing_info_penalty,
            explanation=latest_risk.explanation,
        )
    return PublicAction(
        id=str(action.id),
        title=action.title,
        description=action.description or None,
        deadline=action.deadline,
        severity=action.severity,
        risk_score=latest_risk.score if latest_risk else None,
        risk=risk_breakdown,
        deadline_confidence=(
            action.deadline_confidence if action.deadline_confidence > 0 else None
        ),
        deadline_source=action.deadline_source,
        status=action.status,
        steps=action.steps or [],
        evidence_span=action.evidence_span or None,
        reply_needed=action.reply_needed,
        amount_due_eur=action.amount_due_eur,
    )


def _load_risk_by_action(db: Session, action_ids: list[UUID]) -> dict[UUID, RiskScore]:
    """Batch-load the most recent RiskScore per action — O(1) queries."""
    if not action_ids:
        return {}
    stmt = (
        select(RiskScore)
        .where(RiskScore.action_item_id.in_(action_ids))
        .order_by(RiskScore.computed_at.desc())
    )
    out: dict[UUID, RiskScore] = {}
    for rs in db.scalars(stmt).all():
        # First-seen wins because of ORDER BY DESC.
        out.setdefault(rs.action_item_id, rs)
    return out


def _public_letter(db: Session, letter: Letter) -> PublicLetter:
    actions = list(
        db.scalars(select(ActionItem).where(ActionItem.letter_id == letter.id)).all()
    )
    risk_by_action = _load_risk_by_action(db, [a.id for a in actions])
    # Citations are stored as a list[dict] on the Letter row, but the public
    # contract uses CitationItem. Coerce gracefully — drop malformed entries
    # rather than 500 the whole letter response if the AI produced a string
    # citation or a missing field.
    raw_citations = letter.citations or []
    public_citations: list[CitationItem] = []
    for c in raw_citations:
        if isinstance(c, dict):
            section = str(c.get("section") or "").strip()
            text = str(c.get("text") or "").strip()
            if section:
                public_citations.append(CitationItem(section=section, text=text))
        elif isinstance(c, str) and c.strip():
            # Lenient fallback for AI outputs that emitted bare strings.
            public_citations.append(CitationItem(section=c.strip(), text=""))

    return PublicLetter(
        id=str(letter.id),
        institution=letter.institution,
        document_type=letter.document_type,
        category=letter.category,
        summary_en=letter.summary,  # field renamed for frontend contract
        ocr_text=letter.ocr_text or None,
        confidence=letter.confidence,
        actions=[_public_action(a, risk_by_action.get(a.id)) for a in actions],
        extraction_warnings=letter.extraction_warnings or [],
        explanation=letter.explanation or "",
        consequence=letter.consequence or "",
        risk_reason=letter.risk_reason or "",
        checklist=list(letter.checklist or []),
        citations=public_citations,
        response_draft=letter.response_draft or "",
    )


# ---------- POST /letters — synchronous upload + extract ----------


@router.post(
    "/letters",
    response_model=PublicLetter,
    status_code=200,
    summary="Upload a letter and return the full extraction (synchronous)",
    description=(
        "Frontend-facing root-level route. Combines `POST /api/letters/upload`"
        " + `POST /api/letters/{id}/extract` into one call: persists the file,"
        " runs the Qwen3.7-Plus vision + extraction call, persists the actions"
        " and risk scores, and returns the full PublicLetter inline.\n\n"
        "For streaming progress events use the richer SSE route "
        "`GET /api/letters/{id}/process` instead."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "`LETTER_EMPTY_UPLOAD`."},
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        413: {"model": ErrorResponse, "description": "`LETTER_TOO_LARGE`."},
        415: {
            "model": ErrorResponse,
            "description": "`LETTER_UNSUPPORTED_TYPE` / `LETTER_CORRUPT_FILE` / `LETTER_MIME_MISMATCH`.",
        },
        502: {"model": ErrorResponse, "description": "`EXTRACTION_FAILED`."},
    },
)
async def post_letter(
    file: UploadFile = File(...),
    lang: str | None = Query(default=None, max_length=8),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
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
    db.flush()

    saved_path = save_letter_file(user.id, letter.id, actual_mime, image_bytes)
    letter.original_file = saved_path
    letter.status = LetterStatus.PROCESSING
    db.add(letter)
    db.commit()

    try:
        extracted = await extract_from_letter_file(
            saved_path, actual_mime, lang=out_lang
        )
    except Exception as exc:
        # Log the real Qwen error to the server console so we can diagnose.
        # The 502 response stays generic on the wire to avoid leaking provider
        # implementation details to the client.
        logger.exception(
            "Qwen extraction failed for letter %s: %s",
            letter.id,
            exc,
        )
        letter.status = LetterStatus.ERROR
        db.add(letter)
        db.commit()
        raise KlarHTTPException(502, ErrorCode.EXTRACTION_FAILED)

    persist_extraction(db, letter, extracted)
    letter.status = LetterStatus.COMPLETED
    letter.processed_at = utcnow()
    db.add(letter)
    db.commit()
    db.refresh(letter)

    return _public_letter(db, letter)


# ---------- GET /letters/{id} ----------


@router.get(
    "/letters/{letter_id}",
    response_model=PublicLetter,
    summary="Fetch a single letter in the frontend-facing shape",
    description=(
        "Frontend-facing root-level route. Returns the same `PublicLetter` "
        "as `POST /letters`. The `?lang=` parameter is accepted but does NOT "
        "re-extract — localization happens at extraction time. Re-localization "
        "support is a future enhancement."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
    },
)
def get_letter_public(
    letter_id: UUID,
    lang: str | None = Query(default=None, max_length=8),  # noqa: ARG001 — accepted for contract compat
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)
    return _public_letter(db, letter)


# ---------- GET /actions ----------


@router.get(
    "/actions",
    response_model=list[PublicActionListItem],
    summary="List the current user's action items (home / deadlines / agenda feed)",
    description=(
        "Frontend-facing root-level route. Empty-string `?status=` is "
        "treated as 'no filter' — safe to bind directly to React state. "
        "Unknown values return `VALIDATION_ERROR`."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        422: {"model": ErrorResponse, "description": "Unknown status value."},
    },
)
def list_actions_public(
    status: str | None = Query(default=None),
    lang: str | None = Query(default=None, max_length=8),  # noqa: ARG001 — accepted for contract compat
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    parsed_status: ActionStatus | None = None
    if status:
        try:
            parsed_status = ActionStatus(status)
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
                            + ", ".join(s.value for s in ActionStatus),
                        }
                    ]
                },
            )

    stmt = (
        select(ActionItem)
        .join(Letter, Letter.id == ActionItem.letter_id)
        .where(Letter.user_id == user.id)
    )
    if parsed_status:
        stmt = stmt.where(ActionItem.status == parsed_status)
    items = list(db.scalars(stmt).all())
    return [
        PublicActionListItem(
            id=str(a.id),
            letter_id=str(a.letter_id),
            title=a.title,
            deadline=a.deadline,
            severity=a.severity,
            status=a.status,
            reply_needed=a.reply_needed,
            amount_due_eur=a.amount_due_eur,
        )
        for a in items
    ]


# ---------- PATCH /actions/{id} ----------


@router.patch(
    "/actions/{action_id}",
    response_model=PublicActionUpdateResponse,
    summary="Update an action (mark done / reopen / ignore)",
    description=(
        "Frontend-facing root-level route. Send only the fields you want to "
        "change. Each changed field is logged to `UserCorrection` server-side."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`ACTION_NOT_FOUND`."},
    },
)
def update_action_public(
    action_id: UUID,
    payload: ActionUpdate,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    from app.models import UserCorrection

    item = db.get(ActionItem, action_id)
    if item is None:
        raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)
    letter = db.get(Letter, item.letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)

    changes = payload.model_dump(exclude_unset=True)
    for field, new_value in changes.items():
        old_value = getattr(item, field)
        if old_value != new_value:
            db.add(
                UserCorrection(
                    action_item_id=action_id,
                    field_name=field,
                    original_value=str(old_value),
                    corrected_value=str(new_value),
                )
            )
            setattr(item, field, new_value)

    db.add(item)
    db.commit()
    return PublicActionUpdateResponse(id=str(item.id), status=item.status)


# ---------- POST /letters/{id}/reply ----------


@router.post(
    "/letters/{letter_id}/reply",
    response_model=ReplyDraft,
    summary="Generate a ready-to-send German reply (Behördendeutsch)",
    description=(
        "Produces a formal German Antwortbrief for the institution that sent "
        "this letter. The body is **always in German** regardless of `?lang=` — "
        "the recipient is a German authority. Optional body fields:\n\n"
        "- `action_id` — scope the reply to ONE specific action on the letter\n"
        "- `applicant` — free-form `{field: value}` map (name, address, "
        "Steuernummer, …) woven into the letter header"
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {
            "model": ErrorResponse,
            "description": "`LETTER_NOT_FOUND` or `ACTION_NOT_FOUND`.",
        },
        502: {"model": ErrorResponse, "description": "`LLM_PROVIDER_ERROR`."},
    },
)
async def generate_reply(
    letter_id: UUID,
    payload: ReplyRequest | None = None,
    lang: str | None = Query(default=None, max_length=8),  # noqa: ARG001 — kept for contract compat; body_text is always German
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    payload = payload or ReplyRequest()

    # If action_id is set, validate that it belongs to this letter.
    if payload.action_id:
        try:
            action_uuid = UUID(payload.action_id)
        except ValueError:
            raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)
        action = db.get(ActionItem, action_uuid)
        if action is None or action.letter_id != letter.id:
            raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)

    # 1) Retrieve real legal context from the AI team's law corpus
    try:
        from ai.rag.retrieval import retrieve_legal_context

        # AI team's new signature (commit 61fd2b5): (letter_type, consequence, top_k)
        legal_chunks = retrieve_legal_context(
            letter_type=letter.document_type or letter.letter_type or "",
            consequence=letter.consequence or letter.summary or "",
            top_k=5,
        )
    except Exception as exc:
        # Retrieval is a nice-to-have for citations; never block the reply on it.
        logger.warning("Legal retrieval failed: %s — proceeding without citations", exc)
        legal_chunks = []

    # 2) Synthesize the AgentResult their generator expects from our DB state
    agent_result = ai_bridge.synthesize_agent_result(letter, action=None)

    # If the caller passed applicant data, weave it into the OCR text so the
    # generator's prompt sees it (their prompt has no separate applicant slot).
    enriched_ocr = letter.ocr_text or ""
    if payload.applicant:
        applicant_lines = "\n".join(
            f"  - {k}: {v}" for k, v in payload.applicant.items() if v
        )
        enriched_ocr += f"\n\n[Absender / Applicant details:\n{applicant_lines}\n]"

    # 3) Call the grounded generator (anti-hallucination + real §§ in scope)
    try:
        generation = await ai_bridge.generate_grounded_response(
            ocr_text=enriched_ocr,
            agent_result=agent_result,
            language="de",  # body_text always German per frontend contract
            legal_chunks=legal_chunks,
        )
    except Exception as exc:
        logger.exception(
            "Reply generation failed for letter %s: %s",
            letter_id,
            exc,
        )
        raise KlarHTTPException(502, ErrorCode.LLM_PROVIDER_ERROR)

    # 4) Unpack + persist all 4 long-form fields
    explanation, body_text, checklist, citations = ai_bridge.unpack_generation_output(
        generation
    )
    letter.explanation = explanation
    letter.response_draft = body_text
    letter.checklist = checklist
    letter.citations = citations
    db.add(letter)
    db.commit()

    return ReplyDraft(body_text=body_text, language="de", download_url=None)


# ---------- POST /rag/search ----------


@router.post(
    "/rag/search",
    response_model=RagResponse,
    summary="Grounded-answer search over the German bureaucracy knowledge corpus",
    description=(
        "Frontend-facing root-level route. Body: `{query, top_k?, institution?}`. "
        "Returns `{hits: [{text, score, metadata}]}`. The `text` is German "
        "(it's a legal/institutional reference) — never translated by the API. "
        "Empty `hits: []` is valid (the chat falls back to a generic answer)."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
    },
)
def rag_search_public(
    payload: RagQuery,
    _: User = Depends(get_current_user),
):
    """Frontend-facing RAG search — backed by the AI team's 120k-line
    real German law corpus at `ai/data/chroma/` (see docs/07 §12).
    """
    from ai.rag.retrieval import retrieve_legal_context

    # AI team's new signature: (letter_type, consequence, top_k).
    chunks = retrieve_legal_context(
        letter_type=payload.query,
        consequence=payload.institution or "",
        top_k=payload.top_k,
    )
    return RagResponse(hits=[ai_bridge.legal_chunk_to_rag_hit(c) for c in chunks])


# ---------- POST /chat — letter-aware follow-up assistant ----------


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a follow-up question about a specific letter",
    description=(
        "Grounded chat: retrieves legal context via RAG, combines with the "
        "letter's extracted data, and calls Qwen to produce a concise answer."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
    },
)
async def chat_about_letter(
    payload: ChatRequest,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    import os
    from uuid import UUID as _UUID

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    from ai.prompts import CHAT_SYSTEM_PROMPT

    try:
        letter_uuid = _UUID(payload.letter_id)
    except ValueError:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    letter = db.get(Letter, letter_uuid)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    # Build legal context from what's already stored on the letter
    citations = letter.citations or []
    if citations:
        legal_context = "\n".join(
            f"- {c.get('section', '§')}: {c.get('text', '')}" for c in citations
        )
    else:
        legal_context = "(no legal references available for this letter)"

    system = CHAT_SYSTEM_PROMPT.format(
        institution=letter.institution or "",
        document_type=letter.document_type or "",
        category=letter.category.value if letter.category else "",
        summary=letter.summary or "",
        consequence=letter.consequence or "",
        ocr_text_short=(letter.ocr_text or "")[:1500],
        legal_context=legal_context,
    )

    model = ChatOpenAI(
        model=os.environ.get("QWEN_AGENT_MODEL", "qwen3.7-plus"),
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=os.environ.get(
            "QWEN_API_BASE",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0,
        max_tokens=512,
        extra_body={"enable_thinking": False},
    )

    response = await model.ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=payload.query),
        ]
    )

    # Coerce raw dicts → CitationItem (same as _public_letter)
    clean_citations = []
    for c in citations:
        if isinstance(c, dict) and c.get("section"):
            clean_citations.append(
                CitationItem(
                    section=str(c.get("section", "")),
                    text=str(c.get("text", "")),
                )
            )
    return ChatResponse(answer=response.content, citations=clean_citations)


# ---------- POST /letters/{id}/form-fill — annotated form image ----------


@router.post(
    "/letters/{letter_id}/form-fill",
    summary="Generate the letter image with placeholder annotations for fields to fill",
    description=(
        "Takes the original uploaded letter, identifies fields the user needs to fill, "
        "and returns a PNG image with red placeholder text overlaid on those fields. "
        "Ready to download/print."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
        502: {"model": ErrorResponse, "description": "Image generation failed."},
    },
)
async def form_fill(
    letter_id: UUID,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    from fastapi.responses import Response

    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    if not letter.original_file:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    # Build placeholders from the letter's checklist and actions
    placeholders = []

    # From checklist (documents to prepare)
    for item in letter.checklist or []:
        placeholders.append(str(item))

    # From actions (steps the user needs to take)
    actions = list(
        db.scalars(select(ActionItem).where(ActionItem.letter_id == letter.id)).all()
    )
    for action in actions:
        for step in action.steps or []:
            if any(
                kw in step.lower()
                for kw in [
                    "iban",
                    "steuer",
                    "name",
                    "adresse",
                    "address",
                    "nummer",
                    "number",
                    "unterschrift",
                    "signature",
                    "datum",
                    "date",
                    "versichertennummer",
                    "aktenzeichen",
                ]
            ):
                placeholders.append(step)

    # Always include standard form fields — these match the PLACEHOLDER_MAP
    # in ai/form_fill.py and produce clear English placeholders
    standard_fields = [
        "Steuernummer",
        "Name, Vorname",
        "Anschrift",
        "Telefon / E-Mail",
        "IBAN für Erstattung",
        "Ort, Datum",
        "Unterschrift",
    ]
    # Add standard fields that aren't already covered
    existing_lower = {p.lower() for p in placeholders}
    for f in standard_fields:
        if not any(f.lower() in e for e in existing_lower):
            placeholders.append(f)

    try:
        from ai.form_fill import generate_filled_form

        image_bytes = await generate_filled_form(
            image_path=letter.original_file,
            placeholders=placeholders[:8],  # Cap at 8 to keep prompt manageable
        )
    except Exception as exc:
        logger.exception(
            "Form-fill generation failed for letter %s: %s", letter_id, exc
        )
        raise KlarHTTPException(502, ErrorCode.LLM_PROVIDER_ERROR)

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="klar-form-{str(letter_id)[:8]}.png"',
        },
    )


# ============================================================
# SSE flow — root-level aliases for the 2-step pipeline
# ============================================================
# Frontend's docs/06 contract pins all routes at root (no /api prefix).
# The SSE pipeline natively lives at /api/letters/upload + /api/letters/
# {id}/process. These two routes mirror those at root so the frontend's
# `lib/api/client.ts` can use the SSE flow without rewriting its BASE.


@router.post(
    "/letters/upload",
    response_model=LetterUploadResponse,
    status_code=201,
    summary="(Step 1 of SSE flow) Upload a letter, return its id — no AI yet",
    description=(
        "Root-level alias of `POST /api/letters/upload`. Persists the file "
        "to disk and returns `{letter_id}` immediately (~50ms). The heavy "
        "AI extraction runs LATER via `GET /letters/{id}/process` (SSE)."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "`LETTER_EMPTY_UPLOAD`."},
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        413: {"model": ErrorResponse, "description": "`LETTER_TOO_LARGE`."},
        415: {
            "model": ErrorResponse,
            "description": "`LETTER_UNSUPPORTED_TYPE` / `LETTER_CORRUPT_FILE` / `LETTER_MIME_MISMATCH`.",
        },
    },
)
async def upload_letter_public(
    file: UploadFile = File(...),
    lang: str | None = Query(default=None, max_length=8),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Same impl as /api/letters/upload — file validation, magic-bytes
    check, persist, return {letter_id}.
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
    db.flush()
    saved_path = save_letter_file(user.id, letter.id, actual_mime, image_bytes)
    letter.original_file = saved_path
    db.add(letter)
    db.commit()
    db.refresh(letter)
    return LetterUploadResponse(letter_id=letter.id)


@router.get(
    "/letters/{letter_id}/process",
    summary="(Step 2 of SSE flow) Stream the AI pipeline as Server-Sent Events",
    description=(
        "Root-level alias of `GET /api/letters/{letter_id}/process`. Runs "
        "the AI team's full 4-step pipeline (OCR → ReAct agent → RAG "
        "retrieval → grounded generator) and streams events live:\n\n"
        "  `ocr_result`, `classification`, `risk_score`, `deadline`, "
        "  `consequence`, `explanation` (chunked), `response_draft` "
        "  (chunked), `checklist`, `citations`, `done`\n\n"
        "Frontend opens with `new EventSource(url, { withCredentials: true })` "
        "so the session cookie travels."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`LETTER_NOT_FOUND`."},
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream — see /api/letters/{id}/process for event schemas.",
        },
    },
)
async def process_letter_public(
    letter_id: UUID,
    lang: str | None = Query(default=None, max_length=8),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    letter = db.get(Letter, letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.LETTER_NOT_FOUND)

    out_lang = normalize_lang(lang or letter.language or user.language)
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
