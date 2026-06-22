from dataclasses import dataclass
from pydantic import BaseModel, Field


# --- Structured output models (used by LLM response_format) ---


class Classification(BaseModel):
    type: str = Field(
        description="Letter type, e.g. 'Residence Permit - Document Request', 'Health Insurance - Tax ID Request'"
    )
    agency: str = Field(
        description="Sender agency name, e.g. 'Techniker Krankenkasse', 'Ausländerbehörde München'"
    )


class Deadline(BaseModel):
    date: str | None = Field(
        description="Deadline date in YYYY-MM-DD format, or null if no deadline"
    )
    days_remaining: int | None = Field(
        description="Days until deadline from today, or null"
    )
    source: str = Field(
        description="'letter' if read directly, 'calculated' if computed from letter date, 'searched' if from web, 'none' if no deadline applies"
    )


class Consequence(BaseModel):
    text: str = Field(
        description="Detailed consequence description of what happens if deadline is missed or action not taken"
    )
    severity: str = Field(description="One-line severity summary")


class RiskScore(BaseModel):
    score: int = Field(description="Risk score 1-5", ge=1, le=5)
    label: str = Field(description="Informational, Low, Medium, High, or Critical")
    reason: str = Field(description="Why this risk score was assigned")


class AgentAnalysis(BaseModel):
    """Structured output from the ReAct agent letter analysis."""

    classification: Classification
    deadline: Deadline
    consequence: Consequence
    risk_score: RiskScore


class Citation(BaseModel):
    section: str = Field(
        description="Legal paragraph reference, e.g. '§ 81 Abs. 4 AufenthG'"
    )
    text: str = Field(description="Brief explanation of why this citation is relevant")


class GenerationOutput(BaseModel):
    """Structured output from the response generation LLM."""

    explanation: str = Field(
        description="Clear plain-language explanation of the letter"
    )
    response_draft: str = Field(description="Formal response letter in Behördendeutsch")
    checklist: list[str] = Field(
        description="List of documents the user needs to prepare, with German terms in parentheses"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Legal § references that are relevant. Empty list if none found.",
    )


# --- Internal data transfer objects ---


@dataclass
class AgentEvent:
    type: str  # "classification", "risk_score", "deadline", "consequence", "error"
    data: dict


@dataclass
class AgentResult:
    ocr_text: str
    letter_type: str
    agency: str
    deadline_date: str | None
    days_remaining: int | None
    consequence: str
    risk_score: int
    risk_label: str
