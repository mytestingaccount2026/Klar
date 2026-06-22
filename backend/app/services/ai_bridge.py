"""Adapter module: translate AI team's schemas ↔ ours.

Per `docs/07-pipeline-integration-decisions.md`, the AI team's modules
(`ai/react_agent/*`, `ai/rag/*`) produce shapes that don't match the
frontend's expected wire format. This module is the SINGLE BOUNDARY where
their schemas (`AgentAnalysis`, `Classification`, `Deadline`,
`RiskScore`, `Consequence`, `Citation`, `GenerationOutput`, `LegalChunk`,
`AgentResult`) are translated into ours (`DocumentCategory`, `Severity`,
`DeadlineSource`, `PublicAction`, `RagHit`, etc.).

Nothing in `app/routers/` or `app/pipeline/` should ever import directly
from `ai.*` schemas — always go through this bridge.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from app.models import (
    ActionItem,
    DeadlineSource,
    DocumentCategory,
    Letter,
    Severity,
)
from app.schemas import RagHit

if TYPE_CHECKING:
    # Imported only at type-check time so the runtime cost of importing the
    # whole langchain stack only happens when someone actually USES the bridge.
    from ai.schemas import (
        AgentAnalysis,
        AgentResult,
        Citation,
        GenerationOutput,
    )
    from ai.rag.retrieval import LegalChunk

logger = logging.getLogger("klar.ai_bridge")


# ============================================================
# Their Classification.type (free text) → our DocumentCategory
# ============================================================
#
# Order matters — patterns checked top-to-bottom. First match wins.
# Patterns are case-insensitive substring matches against `Classification.type`.

_CATEGORY_PATTERNS: list[tuple[str, DocumentCategory]] = [
    # Most specific first
    ("residence permit", DocumentCategory.IMMIGRATION),
    ("aufenthaltstitel", DocumentCategory.IMMIGRATION),
    ("ausländerbehörde", DocumentCategory.IMMIGRATION),
    ("visa", DocumentCategory.IMMIGRATION),
    ("aufenthalts", DocumentCategory.IMMIGRATION),
    ("immigration", DocumentCategory.IMMIGRATION),
    ("health insurance", DocumentCategory.HEALTH_INSURANCE),
    ("krankenkasse", DocumentCategory.HEALTH_INSURANCE),
    ("krankenversicherung", DocumentCategory.HEALTH_INSURANCE),
    ("aok", DocumentCategory.HEALTH_INSURANCE),
    ("techniker krankenkasse", DocumentCategory.HEALTH_INSURANCE),
    ("barmer", DocumentCategory.HEALTH_INSURANCE),
    ("dak-gesundheit", DocumentCategory.HEALTH_INSURANCE),
    ("car insurance", DocumentCategory.OTHER_INSURANCE),
    ("haftpflicht", DocumentCategory.OTHER_INSURANCE),
    ("hausrat", DocumentCategory.OTHER_INSURANCE),
    ("kfz-versicherung", DocumentCategory.OTHER_INSURANCE),
    ("liability insurance", DocumentCategory.OTHER_INSURANCE),
    ("tax", DocumentCategory.TAX),
    ("finanzamt", DocumentCategory.TAX),
    ("steuer", DocumentCategory.TAX),
    ("university", DocumentCategory.EDUCATION),
    ("universität", DocumentCategory.EDUCATION),
    ("hochschule", DocumentCategory.EDUCATION),
    ("immatrikulation", DocumentCategory.EDUCATION),
    ("studentenwerk", DocumentCategory.EDUCATION),
    ("bafög", DocumentCategory.EDUCATION),
    ("rückmeldung", DocumentCategory.EDUCATION),
    ("enrollment", DocumentCategory.EDUCATION),
    ("rent", DocumentCategory.HOUSING),
    ("vermieter", DocumentCategory.HOUSING),
    ("hausverwaltung", DocumentCategory.HOUSING),
    ("mieterhöhung", DocumentCategory.HOUSING),
    ("nebenkosten", DocumentCategory.HOUSING),
    ("landlord", DocumentCategory.HOUSING),
    ("electricity", DocumentCategory.UTILITIES),
    ("gas bill", DocumentCategory.UTILITIES),
    ("internet", DocumentCategory.UTILITIES),
    ("telekom", DocumentCategory.UTILITIES),
    ("vodafone", DocumentCategory.UTILITIES),
    ("stadtwerke", DocumentCategory.UTILITIES),
    ("vattenfall", DocumentCategory.UTILITIES),
    ("strom", DocumentCategory.UTILITIES),
    ("employer", DocumentCategory.EMPLOYMENT),
    ("arbeitgeber", DocumentCategory.EMPLOYMENT),
    ("lohn", DocumentCategory.EMPLOYMENT),
    ("gehalt", DocumentCategory.EMPLOYMENT),
    ("payroll", DocumentCategory.EMPLOYMENT),
    ("arbeitsvertrag", DocumentCategory.EMPLOYMENT),
    ("unemployment", DocumentCategory.GOVERNMENT_BENEFITS),
    ("kindergeld", DocumentCategory.GOVERNMENT_BENEFITS),
    ("elterngeld", DocumentCategory.GOVERNMENT_BENEFITS),
    ("wohngeld", DocumentCategory.GOVERNMENT_BENEFITS),
    ("arbeitslosengeld", DocumentCategory.GOVERNMENT_BENEFITS),
    ("bürgergeld", DocumentCategory.GOVERNMENT_BENEFITS),
    ("jobcenter", DocumentCategory.GOVERNMENT_BENEFITS),
    ("familienkasse", DocumentCategory.GOVERNMENT_BENEFITS),
    ("pension", DocumentCategory.PENSION),
    ("rentenversicherung", DocumentCategory.PENSION),
    ("rente", DocumentCategory.PENSION),
    ("rundfunk", DocumentCategory.BROADCAST_FEE),
    ("gez", DocumentCategory.BROADCAST_FEE),
    ("beitragsservice", DocumentCategory.BROADCAST_FEE),
    ("broadcasting fee", DocumentCategory.BROADCAST_FEE),
    ("bürgeramt", DocumentCategory.CIVIC),
    ("einwohnermelde", DocumentCategory.CIVIC),
    ("standesamt", DocumentCategory.CIVIC),
    ("personalausweis", DocumentCategory.CIVIC),
    ("reisepass", DocumentCategory.CIVIC),
    ("meldebescheinigung", DocumentCategory.CIVIC),
    ("court", DocumentCategory.LEGAL_DEBT),
    ("gericht", DocumentCategory.LEGAL_DEBT),
    ("mahnbescheid", DocumentCategory.LEGAL_DEBT),
    ("vollstreckung", DocumentCategory.LEGAL_DEBT),
    ("inkasso", DocumentCategory.LEGAL_DEBT),
    ("bußgeld", DocumentCategory.LEGAL_DEBT),
    ("anwalt", DocumentCategory.LEGAL_DEBT),
    ("debt collection", DocumentCategory.LEGAL_DEBT),
    ("fine notice", DocumentCategory.LEGAL_DEBT),
    ("bank", DocumentCategory.BANKING),
    ("sparkasse", DocumentCategory.BANKING),
    ("schufa", DocumentCategory.BANKING),
    ("kreditkarte", DocumentCategory.BANKING),
]


def map_classification_to_category(free_text_type: str | None) -> DocumentCategory:
    """Map their `Classification.type` free-text string → our closed enum.

    Falls back to `DocumentCategory.OTHER` if no pattern matches.
    """
    if not free_text_type:
        return DocumentCategory.OTHER
    needle = free_text_type.lower()
    for pattern, cat in _CATEGORY_PATTERNS:
        if pattern in needle:
            return cat
    logger.debug(
        "map_classification_to_category: no match for %r → OTHER", free_text_type
    )
    return DocumentCategory.OTHER


# ============================================================
# Their RiskScore.label → our Severity enum
# ============================================================

_LABEL_TO_SEVERITY: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "informational": Severity.LOW,
}


def map_their_severity_label(label: str | None) -> Severity:
    """Map their `RiskScore.label` → our `Severity` enum.

    Per docs/07 §6: their 1-5 score is discarded, only the qualitative label
    survives because we recompute the numerical score via our deterministic
    formula. Unknown labels fall back to MEDIUM.
    """
    if not label:
        return Severity.MEDIUM
    return _LABEL_TO_SEVERITY.get(label.strip().lower(), Severity.MEDIUM)


# ============================================================
# Their Deadline.source → our DeadlineSource enum
# ============================================================

_SOURCE_MAPPING: dict[str, DeadlineSource] = {
    "letter": DeadlineSource.EXPLICIT,
    "calculated": DeadlineSource.INFERRED,
    "searched": DeadlineSource.INFERRED,  # Tavily web search
    "none": DeadlineSource.UNKNOWN,
}


def map_their_deadline_source(source: str | None) -> DeadlineSource:
    """Map their `Deadline.source` (letter|calculated|searched|none) → ours."""
    if not source:
        return DeadlineSource.UNKNOWN
    return _SOURCE_MAPPING.get(source.strip().lower(), DeadlineSource.UNKNOWN)


# A separate "was Tavily used" flag, so the UI can render
# "We searched online to find this deadline" badge.
def deadline_was_web_searched(source: str | None) -> bool:
    return (source or "").strip().lower() == "searched"


# ============================================================
# Our risk_score (0-100) → their RiskScore.label (for generator input)
# ============================================================


def risk_label_from_score(score: int | None) -> str:
    """Map our 0-100 score → their label string (used when synthesizing an
    `AgentResult` to feed their `generate_response`)."""
    s = score or 0
    if s >= 80:
        return "Critical"
    if s >= 60:
        return "High"
    if s >= 40:
        return "Medium"
    if s >= 20:
        return "Low"
    return "Informational"


# ============================================================
# Letter + (optional) ActionItem → AgentResult (their dataclass)
# ============================================================


def synthesize_agent_result(
    letter: Letter,
    action: ActionItem | None = None,
) -> "AgentResult":
    """Build a synthetic `AgentResult` from our persisted Letter + Action.

    Used to feed `ai.rag.generator.generate_response()` after the sync
    extraction path has already run (e.g. for `POST /letters/{id}/reply`,
    where we already have all the structured data and just want a
    grounded German Behördendeutsch draft on top of it).
    """
    from ai.schemas import AgentResult  # lazy import — runtime only when called

    deadline_date_str: str | None = None
    days_remaining: int | None = None

    primary_deadline = (action.deadline if action else None) or letter.deadline_date
    if primary_deadline is not None:
        deadline_date_str = primary_deadline.isoformat()
        days_remaining = (primary_deadline - date.today()).days

    return AgentResult(
        ocr_text=letter.ocr_text or "",
        letter_type=letter.document_type or letter.letter_type or "",
        agency=letter.institution or "",
        deadline_date=deadline_date_str,
        days_remaining=days_remaining,
        consequence=letter.consequence or "",
        risk_score=letter.risk_score or 0,
        risk_label=risk_label_from_score(letter.risk_score),
    )


# ============================================================
# Their LegalChunk → our RagHit (for /rag/search response)
# ============================================================


def legal_chunk_to_rag_hit(chunk: "LegalChunk") -> RagHit:
    """Map their `LegalChunk` from `ai.rag.retrieval` → our `RagHit` wire shape.

    `score` defaults to `1.0` because their retrieval doesn't expose a
    similarity score externally. Section/law/title/citation all surface in
    `metadata` so the frontend can render `§ 81 AufenthG` style citations.
    """
    return RagHit(
        text=chunk.text,
        score=1.0,
        metadata={
            "section": chunk.section,
            "law": chunk.law,
            "title": chunk.title,
            "citation": chunk.citation,
        },
    )


# ============================================================
# Their Citation → JSON dict (stored on Letter.citations column)
# ============================================================


def citation_to_dict(c: "Citation") -> dict[str, Any]:
    """Persist-shape for the `Letter.citations` JSON column.

    Matches `SSECitationItem` schema (`{section, text, score}`) so SSE
    events and the persisted column carry identical shapes.
    """
    return {
        "section": c.section,
        "text": c.text,
        "score": 1.0,  # their structured citation doesn't carry a score
    }


def citations_to_dicts(cits: list["Citation"]) -> list[dict[str, Any]]:
    return [citation_to_dict(c) for c in cits or []]


# ============================================================
# Their GenerationOutput → (explanation, response_draft, checklist[], citations[dict])
# ============================================================


def unpack_generation_output(
    out: "GenerationOutput",
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    """Pure unpacking convenience — returns the 4 tuples each going into a
    separate Letter column.
    """
    return (
        out.explanation or "",
        out.response_draft or "",
        list(out.checklist or []),
        citations_to_dicts(out.citations),
    )


# ============================================================
# Their AgentAnalysis → (category, document_type, severity, deadline_date, ...)
# ============================================================


async def generate_grounded_response(
    ocr_text: str,
    agent_result: "AgentResult",
    language: str = "en",
    legal_chunks: list["LegalChunk"] | None = None,
) -> "GenerationOutput":
    """Grounded variant of `ai.rag.generator.generate_response()`.

    Their stock generator hardcodes `legal_context="[NO LEGAL REFERENCES
    LOADED]"` in the prompt template — meaning the anti-hallucination
    rules tell the model to refuse to cite any §. This wrapper retrieves
    legal context first and injects it into the SAME prompt template so
    the model CAN cite real §§ from the corpus.

    Why we bypass `_model.with_structured_output()`: the model often
    emits `citations: ["§ 81 AufenthG"]` (strings) instead of the
    declared `[{section, text}]` schema. Langchain's strict Pydantic
    parser refuses to coerce. We use raw JSON mode + a lenient parser
    that wraps bare strings as `{section: str, text: ""}`.

    Falls back gracefully when `legal_chunks` is empty.
    """
    import json
    import os
    from ai.rag.generator import LANGUAGE_NAMES
    from ai.prompts import GENERATION_PROMPT
    from ai.schemas import Citation, GenerationOutput
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    # Build the legal-context section from retrieved chunks
    if legal_chunks:
        legal_lines = [
            f"### {c.citation} — {c.title}\n{c.text}\n" for c in legal_chunks
        ]
        legal_context = "\n".join(legal_lines)
    else:
        legal_context = (
            "[NO LEGAL REFERENCES LOADED — do NOT cite any § numbers unless "
            "you are 100% certain they exist. Prefer explaining without "
            "citations over citing something that might be wrong.]"
        )

    prompt = GENERATION_PROMPT.format(
        ocr_text=ocr_text[:3000],
        letter_type=agent_result.letter_type,
        agency=agent_result.agency,
        deadline_date=agent_result.deadline_date or "Not specified",
        days_remaining=agent_result.days_remaining or "Unknown",
        risk_score=agent_result.risk_score,
        risk_label=agent_result.risk_label,
        consequence=agent_result.consequence,
        legal_context=legal_context,
        language=LANGUAGE_NAMES.get(language, "English"),
    )

    # Use raw JSON mode (not structured-output) so we control the parse
    raw_model = ChatOpenAI(
        model=os.environ.get("QWEN_AGENT_MODEL", "qwen3.7-plus"),
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=os.environ.get(
            "QWEN_API_BASE",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0,
        max_tokens=4096,
        extra_body={
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        },
    )
    response = await raw_model.ainvoke([HumanMessage(content=prompt)])
    raw_text = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw_text, list):
        # Some langchain versions return content as a list of parts
        raw_text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_text
        )

    payload = json.loads(raw_text)

    # Normalize citations: accept either list[str] or list[dict] or list[Citation-like]
    raw_citations = payload.get("citations") or []
    cleaned_citations: list[Citation] = []
    for c in raw_citations:
        if isinstance(c, str):
            # Bare "§ 81 AufenthG" — wrap as Citation with empty explanation
            cleaned_citations.append(Citation(section=c, text=""))
        elif isinstance(c, dict):
            cleaned_citations.append(
                Citation(
                    section=str(c.get("section") or c.get("§") or "§"),
                    text=str(c.get("text") or c.get("explanation") or ""),
                )
            )
        else:
            logger.debug("Skipping unparseable citation: %r", c)

    return GenerationOutput(
        explanation=str(payload.get("explanation") or ""),
        response_draft=str(payload.get("response_draft") or ""),
        checklist=[str(x) for x in (payload.get("checklist") or [])],
        citations=cleaned_citations,
    )


def unpack_agent_analysis(
    analysis: "AgentAnalysis",
) -> dict[str, Any]:
    """Spread their AgentAnalysis fields out for the SSE orchestrator.

    Returns a flat dict so caller can pick which pieces to emit / persist.
    Note we DISCARD their `RiskScore.score` (1-5) and use only `.label`
    because we recompute the 0-100 score via our deterministic formula
    (`app.services.risk.compute_risk`).
    """
    from datetime import date as _date

    deadline_iso = analysis.deadline.date
    parsed_deadline: _date | None = None
    if deadline_iso:
        try:
            parsed_deadline = _date.fromisoformat(deadline_iso)
        except ValueError:
            logger.debug(
                "Their agent returned non-ISO deadline %r — dropping", deadline_iso
            )

    return {
        "category": map_classification_to_category(analysis.classification.type),
        "document_type": analysis.classification.type or "",
        "institution": analysis.classification.agency or "",
        "deadline": parsed_deadline,
        "deadline_source": map_their_deadline_source(analysis.deadline.source),
        "deadline_was_searched": deadline_was_web_searched(analysis.deadline.source),
        "consequence": analysis.consequence.text or "",
        "severity": map_their_severity_label(analysis.risk_score.label),
        "risk_label": analysis.risk_score.label or "Medium",
        "risk_reason": analysis.risk_score.reason or "",
    }
