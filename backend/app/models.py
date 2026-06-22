"""SQLModel tables — mirrors DeadlinePivot PRD §7."""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Naive UTC `datetime`. Replacement for `datetime.utcnow()` which is
    removed in Python 3.14. Stored values stay naive (SQLite convention) but
    are produced from a timezone-aware now() to dodge the deprecation."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DeadlineSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(str, Enum):
    OPEN = "open"
    DONE = "done"
    IGNORED = "ignored"


class LetterStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class SupportedLanguage(str, Enum):
    EN = "en"
    DE = "de"


class DocumentCategory(str, Enum):
    """Closed vocabulary for letter classification.

    Covers the institutional landscape for both international students and
    working professionals in Germany. Use OTHER only when the letter genuinely
    does not fit any defined bucket.
    """

    HEALTH_INSURANCE = "health_insurance"  # AOK, TK, BARMER, private KV
    OTHER_INSURANCE = "other_insurance"  # Haftpflicht, Hausrat, KFZ, Leben
    BANKING = "banking"  # bank accounts, credit cards, SCHUFA
    TAX = "tax"  # Finanzamt
    IMMIGRATION = "immigration"  # Ausländerbehörde, residence/visa
    EDUCATION = "education"  # universities, BAföG, Studentenwerk
    HOUSING = "housing"  # landlord, property management
    UTILITIES = "utilities"  # Strom, Gas, Wasser, Internet, Mobilfunk
    EMPLOYMENT = "employment"  # Arbeitgeber, HR, Lohn
    GOVERNMENT_BENEFITS = (
        "government_benefits"  # ALG I/II, Kindergeld, Elterngeld, Wohngeld
    )
    PENSION = "pension"  # Deutsche Rentenversicherung
    BROADCAST_FEE = "broadcast_fee"  # Beitragsservice / Rundfunk
    CIVIC = "civic"  # Bürgeramt, Personalausweis, Pass
    LEGAL_DEBT = "legal_debt"  # Mahnbescheid, Inkasso, Bußgeld, Anwalt
    OTHER = "other"


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    password_hash: str = Field(default="")
    language: str = Field(default="en")
    timezone: str = "Europe/Berlin"
    calendar_connected: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class Session(SQLModel, table=True):
    """Server-side session — cookie carries the token; secret never leaves DB."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    token: str = Field(unique=True, index=True)
    expires_at: datetime
    created_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)
    user_agent: str = ""
    ip_address: str = ""


class PasswordResetToken(SQLModel, table=True):
    """Single-use token issued by /api/auth/forgot, consumed by /api/auth/reset."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    token: str = Field(unique=True, index=True)
    expires_at: datetime
    used_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class Letter(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id", index=True)

    # File location
    original_file: str = ""

    # Spec-flat structured fields (denormalized from ActionItem for /api/letters)
    letter_type: str = ""  # alias of document_type for spec compat
    risk_score: int = 0  # denormalized highest action risk
    deadline_date: Optional[date] = None  # denormalized most-urgent action deadline

    # Rich Klar extras
    institution: str = ""
    document_type: str = ""
    category: DocumentCategory = DocumentCategory.OTHER
    summary: str = ""  # language matches Letter.language
    language: str = "en"

    # OCR + long-form generation outputs
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    # Overall extraction confidence (0..1) — frontend uses <0.85 to surface a
    # "low confidence, get a human" prompt. Computed from extraction outputs.
    confidence: Optional[float] = None
    explanation: str = ""
    response_draft: str = ""  # ALWAYS German (formal reply to German institution)
    checklist: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    citations: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    consequence: str = ""
    # AI agent's narrative reason for the risk score (e.g. "Deadline was 5 years
    # in the past; missing identity hearings can lead to asylum rejection").
    # Distinct from RiskScore.explanation, which is our deterministic-formula
    # breakdown (factors and weights). Both can coexist — they serve different
    # UI surfaces ("why is this critical" vs. "how the score was computed").
    risk_reason: str = ""
    extraction_warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Processing lifecycle
    status: LetterStatus = LetterStatus.UPLOADED
    processed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class ActionItem(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    letter_id: UUID = Field(foreign_key="letter.id", index=True)
    title: str
    description: str = ""
    steps: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    deadline: Optional[date] = None
    deadline_confidence: float = 0.0
    deadline_source: DeadlineSource = DeadlineSource.UNKNOWN
    status: ActionStatus = ActionStatus.OPEN
    severity: Severity = Severity.MEDIUM
    reply_needed: bool = False
    calendar_synced: bool = False
    evidence_span: str = ""
    # Outstanding amount the action requires the user to pay (EUR).
    # Populated by the OCR-text amount extractor in services/amounts.py.
    amount_due_eur: Optional[float] = None


class RiskScore(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    action_item_id: UUID = Field(foreign_key="actionitem.id", index=True)
    score: int = 0
    deadline_proximity_pts: float = 0.0
    institution_weight: float = 0.0
    severity_pts: float = 0.0
    missing_info_penalty: float = 0.0
    explanation: str = ""
    computed_at: datetime = Field(default_factory=utcnow)


class UserCorrection(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    action_item_id: UUID = Field(foreign_key="actionitem.id", index=True)
    field_name: str
    original_value: str
    corrected_value: str
    created_at: datetime = Field(default_factory=utcnow)
