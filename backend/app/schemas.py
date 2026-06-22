"""Pydantic request/response schemas (not tied to DB tables).

This module is the source of truth for what crosses the wire. Everything that
appears in the OpenAPI spec (and thus in the Posting collection / frontend
TypeScript types) comes from here.

Grouped sections:
- Extraction internals (ExtractedLetter / ExtractedAction)
- Public response shapes (LetterResponse, LetterListItem, DeadlineItem, ...)
- Auth response shapes
- SSE event payloads (for documentation of the /process stream)
- RAG search shapes
- Error envelope (shape of every non-2xx response — single source of truth)
"""

from datetime import date, datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from app.models import (
    ActionStatus,
    DeadlineSource,
    DocumentCategory,
    LetterStatus,
    Severity,
)


class ExtractedAction(BaseModel):
    title: str
    description: str = ""
    deadline_iso: Optional[date] = None
    deadline_confidence: float = 0.0
    deadline_source: DeadlineSource = DeadlineSource.UNKNOWN
    severity: Severity = Severity.MEDIUM
    steps: list[str] = Field(default_factory=list)
    reply_needed: bool = False
    evidence_span: str = ""


class ExtractedLetter(BaseModel):
    document_type: str = ""
    institution: str = ""
    category: DocumentCategory = DocumentCategory.OTHER
    category_confidence: float = 0.0
    language_confidence: float = 0.0
    summary: str = ""
    ocr_text: str = ""
    actions: list[ExtractedAction] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class LetterListItem(BaseModel):
    """Compact shape used by GET /api/letters list endpoint."""

    id: UUID
    letter_type: str
    category: DocumentCategory
    risk_score: int
    deadline_date: Optional[date] = None
    status: LetterStatus
    created_at: datetime


class LetterResponse(BaseModel):
    id: UUID
    institution: str
    document_type: str
    letter_type: str
    category: DocumentCategory
    summary: str
    language: str
    risk_score: int
    deadline_date: Optional[date] = None
    explanation: str = ""
    response_draft: str = ""
    checklist: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    consequence: str = ""
    status: LetterStatus
    processed_at: Optional[datetime] = None
    created_at: datetime
    actions: list[dict] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class LetterUploadResponse(BaseModel):
    letter_id: UUID


class ActionUpdate(BaseModel):
    status: Optional[ActionStatus] = None
    deadline: Optional[date] = None
    title: Optional[str] = None
    description: Optional[str] = None


class DeadlineItem(BaseModel):
    """Spec-shaped /api/deadlines item — actually a view over ActionItem."""

    id: UUID
    letter_id: UUID
    title: str
    due_date: date
    status: ActionStatus
    risk_score: int
    severity: Severity
    category: DocumentCategory


class RagQuery(BaseModel):
    query: str
    top_k: int = 4
    institution: Optional[str] = None


class RagHit(BaseModel):
    text: str
    score: float
    metadata: dict


class RagResponse(BaseModel):
    hits: list[RagHit]


class ChatRequest(BaseModel):
    query: str
    letter_id: str


class CitationItem(BaseModel):
    """A single legal citation surfaced by the grounded generator.

    `section` is the German legal reference (e.g. "§ 16 AsylG"); `text`
    explains how that section applies to *this* letter. Both fields are
    safe to render in the localized UI — `section` is itself German
    typography ("§") but is treated as a verbatim identifier, not localized.
    """

    section: str = Field(description="Legal section reference, e.g. '§ 16 AsylG'.")
    text: str = Field(description="Why this section applies to the letter.")


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)


# ===================================================================
# Frontend-facing "public" shapes (root-level routes in app/routers/public.py)
# ===================================================================
# These intentionally match `docs/06-frontend-integration-contract.md` field
# for field so the frontend's TypeScript types deserialize cleanly. They are a
# narrower view of the full LetterResponse — extra Klar columns are omitted.


class RiskBreakdown(BaseModel):
    """The factors behind risk_score (mirrors `RiskScore` DB row).

    Powers the "why this risk" detail view in the frontend.
    """

    score: int = Field(ge=0, le=100, description="Identical to risk_score.")
    deadline_proximity_pts: float = Field(ge=0.0, le=1.0)
    institution_weight: float = Field(ge=0.0, le=1.0)
    severity_pts: float = Field(ge=0.0, le=1.0)
    missing_info_penalty: float = Field(ge=0.0, le=1.0)
    explanation: str


class PublicAction(BaseModel):
    """Action shape exposed to the frontend on the root-level routes."""

    id: str
    title: str
    description: Optional[str] = None
    deadline: Optional[date] = Field(default=None, description="YYYY-MM-DD or null")
    severity: Severity
    risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    risk: Optional[RiskBreakdown] = Field(
        default=None,
        description="Full RiskScore breakdown — powers the 'why this risk' view.",
    )
    deadline_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0..1 confidence in the deadline value (null when unknown).",
    )
    deadline_source: Optional[DeadlineSource] = Field(
        default=None,
        description="explicit | inferred | unknown",
    )
    status: ActionStatus = ActionStatus.OPEN
    steps: list[str] = Field(default_factory=list)
    evidence_span: Optional[str] = Field(
        default=None,
        description="Exact German source sentence — never localized.",
    )
    reply_needed: bool = False
    amount_due_eur: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Outstanding amount the user must pay for this action, in EUR. "
            "Extracted from the OCR text by a regex pattern matcher."
        ),
    )


class PublicLetter(BaseModel):
    """Letter shape exposed to the frontend on the root-level routes.

    `summary_en` is the localized summary (the trailing `_en` is a historical
    name in the frontend contract — value is in whatever ?lang= was requested).
    """

    id: str
    institution: str = Field(description="German verbatim — never localized.")
    document_type: str = Field(description="German verbatim — never localized.")
    category: DocumentCategory
    summary_en: str = Field(
        description="Localized summary (field name kept for frontend compat)."
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Verbatim German OCR text from the source. Never localized.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0..1 overall extraction confidence. <0.85 triggers a 'get a human' UI prompt.",
    )
    actions: list[PublicAction] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    # ----- Long-form generated content -----
    # Populated by the SSE pipeline (RAG-grounded generator). Empty strings /
    # empty lists are used as "not generated" sentinels rather than null so
    # frontend code never has to null-guard before iterating.
    explanation: str = Field(
        default="",
        description=(
            "Long-form, plain-language explanation of the letter. Localized to "
            "?lang=. Cite-grounded against German legal sections in `citations`."
        ),
    )
    consequence: str = Field(
        default="",
        description="Short narrative of what happens if the user ignores this letter.",
    )
    risk_reason: str = Field(
        default="",
        description=(
            "AI agent's narrative justification for the risk score. Distinct "
            "from the deterministic factor breakdown on each action's `risk`."
        ),
    )
    checklist: list[str] = Field(
        default_factory=list,
        description="Documents / items the user should prepare. Each line is a single bullet.",
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Legal sections cited by the explanation, in display order.",
    )
    response_draft: str = Field(
        default="",
        description=(
            "Pre-drafted Behördendeutsch reply. ALWAYS in formal German "
            "regardless of `?lang=` — the letter is addressed to a German "
            "institution."
        ),
    )


class PublicActionListItem(BaseModel):
    """Row shape for GET /actions feed (home / deadlines / agenda)."""

    id: str
    letter_id: str
    title: str
    deadline: Optional[date] = None
    severity: Severity
    status: ActionStatus
    reply_needed: bool
    amount_due_eur: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Outstanding EUR amount for this action, mirrored from the same "
            "field on PublicAction. Included on the list shape so the "
            "deadlines/agenda page can show a 'total outstanding' tile "
            "without round-tripping to fetch every parent letter."
        ),
    )


class PublicActionUpdateResponse(BaseModel):
    """Response from PATCH /actions/{id} — minimal echo."""

    id: str
    status: ActionStatus


# ---------- Reply generation ----------


class ReplyRequest(BaseModel):
    """Body for POST /letters/{id}/reply.

    Both fields are optional. `applicant` is a free-form `{field: value}` map
    that the model weaves into the letter (name, address, Steuernummer, etc.).
    """

    action_id: Optional[str] = Field(
        default=None,
        description="Scope the reply to one specific action, if the letter has multiple.",
    )
    applicant: Optional[dict] = Field(
        default=None,
        description="Free-form applicant details: {name, address, ...}. Woven into the letter.",
    )


class ReplyDraft(BaseModel):
    """Response from POST /letters/{id}/reply.

    `body_text` is ALWAYS in formal German regardless of `?lang=` — the letter
    is addressed to a German institution. `download_url` is optional; null means
    the frontend should render PDF / .txt client-side.
    """

    body_text: str = Field(description="Ready-to-send Behördendeutsch.")
    language: str = Field(default="de", description="Always 'de'.")
    download_url: Optional[str] = Field(
        default=None,
        description="Optional server-rendered PDF URL. Null if not available.",
    )


# ===================================================================
# Auth response shapes
# ===================================================================


class UserPublic(BaseModel):
    """Public profile of the authenticated user. Never exposes password_hash."""

    id: str
    email: EmailStr
    language: str = Field(description="ISO 639-1 short code: 'en' or 'de'.")
    timezone: str = Field(description="IANA timezone, e.g. 'Europe/Berlin'.")
    created_at: datetime


class AuthResponse(BaseModel):
    """Successful signup or login. Cookie is set by the server."""

    user: UserPublic


class OkResponse(BaseModel):
    """Generic ok envelope for endpoints with no body to return."""

    ok: bool = True


class ForgotPasswordResponse(BaseModel):
    """Identical shape for known and unknown emails — no enumeration.

    In dev mode (`DEV_AUTH_EXPOSE_RESET_TOKEN=true`) the reset token is
    included so the frontend can drive end-to-end testing without SMTP.
    """

    ok: bool = True
    message: str
    dev_reset_token: Optional[str] = Field(
        default=None,
        description="Dev-only: present only when DEV_AUTH_EXPOSE_RESET_TOKEN=true.",
    )
    dev_expires_at: Optional[datetime] = None


# ===================================================================
# Error envelope (shape of every non-2xx response)
# ===================================================================


class ValidationFieldError(BaseModel):
    field: str = Field(description="Dotted path to the offending field.")
    message: str


class ErrorDetails(BaseModel):
    """Optional structured extras attached to an error envelope."""

    errors: Optional[list[ValidationFieldError]] = None
    declared: Optional[str] = None
    detected: Optional[str] = None


class ErrorResponse(BaseModel):
    """Every non-2xx response from this API has this shape.

    Frontend code should switch on `code` (stable identifier), display
    `message` (or `detail`, same value) to the user, and inspect `details`
    for field-level information on `VALIDATION_ERROR`s.

    `detail` is a wire-compatibility alias for `message` — kept so clients
    written against the FastAPI default `{"detail": "..."}` shape (the
    `06-frontend-integration-contract.md` clients) keep working without
    code changes.
    """

    code: str = Field(
        description="Stable machine-readable identifier — see docs/06-api-contract.md."
    )
    message: str = Field(description="Localized, user-facing copy.")
    detail: str = Field(
        description="Alias of `message` — for clients that expect FastAPI's default error shape."
    )
    details: Optional[ErrorDetails] = None


# ===================================================================
# SSE event payloads (documentation for /api/letters/{id}/process)
# ===================================================================
# The SSE stream is text/event-stream — these schemas describe the JSON
# payload inside each `data:` frame. Frontend can use them to type the
# EventSource handlers.


class SSEOcrResult(BaseModel):
    text: str = Field(description="Verbatim German OCR'd text.")


class SSEClassification(BaseModel):
    type: str = Field(description="Specific German document name, e.g. 'Mahnung'.")
    category: DocumentCategory
    agency: str = Field(description="Sender institution as printed on the letter.")
    category_confidence: float = Field(ge=0.0, le=1.0)


class SSERiskScore(BaseModel):
    score: int = Field(ge=0, le=100)
    label: Literal["Critical", "High", "Medium", "Low"]


class SSEDeadline(BaseModel):
    date: Optional[date] = None
    days_remaining: Optional[int] = None
    note: Optional[str] = None


class SSEConsequence(BaseModel):
    text: str


class SSEStreamChunk(BaseModel):
    """Used by `explanation` and `response_draft` events — one token chunk."""

    chunk: str


class SSEChecklist(BaseModel):
    items: list[str]


class SSECitationItem(BaseModel):
    section: str
    text: str
    score: float


class SSECitations(BaseModel):
    items: list[SSECitationItem]


class SSEDone(BaseModel):
    letter_id: UUID


class SSEErrorPayload(BaseModel):
    """Error payload inside an SSE `data:` frame — same shape as ErrorResponse."""

    code: str
    message: str
    details: Optional[ErrorDetails] = None
