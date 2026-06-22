"""Persistence helper: write an ExtractedLetter into the SQLite tables.

Lives in `services/` so it can be reused by both `routers/letters.py`
(synchronous extract endpoint) and `pipeline/orchestrator.py` (SSE stream)
without inducing a router→pipeline→router import cycle.
"""

from sqlmodel import Session as DBSession

from app.models import ActionItem, Letter, RiskScore
from app.schemas import ExtractedLetter
from app.services.amounts import primary_outstanding_amount
from app.services.risk import compute_risk


def persist_extraction(
    db: DBSession, letter: Letter, extracted: ExtractedLetter
) -> list[ActionItem]:
    """Mutate `letter` + create child ActionItem / RiskScore rows.

    Caller is responsible for committing the surrounding transaction. The
    function does call `db.flush()` to obtain ActionItem IDs for the RiskScore
    foreign key but never commits — that lets the caller batch this into a
    larger transaction (e.g. a single SSE step).
    """
    letter.institution = extracted.institution
    letter.document_type = extracted.document_type
    letter.letter_type = extracted.document_type
    letter.category = extracted.category
    letter.summary = extracted.summary
    letter.ocr_text = extracted.ocr_text
    letter.extraction_warnings = extracted.extraction_warnings

    # Overall confidence = min of available signals. Frontend uses <0.85 to
    # surface a "get a human" prompt.
    signals = [
        s
        for s in (extracted.language_confidence, extracted.category_confidence)
        if s and s > 0
    ]
    letter.confidence = min(signals) if signals else None

    # Detect outstanding €amount from the OCR text once per letter; attach to
    # whichever action carries the highest severity (most-likely the one
    # that triggers the payment). Avoid spreading the same amount across
    # every action — that would double-count in frontend totals.
    letter_amount = primary_outstanding_amount(extracted.ocr_text)
    amount_attached_to: ActionItem | None = None

    saved: list[ActionItem] = []
    highest_score = 0
    earliest_deadline = None

    for a in extracted.actions:
        item = ActionItem(
            letter_id=letter.id,
            title=a.title,
            description=a.description,
            steps=a.steps,
            deadline=a.deadline_iso,
            deadline_confidence=a.deadline_confidence,
            deadline_source=a.deadline_source,
            severity=a.severity,
            reply_needed=a.reply_needed,
            evidence_span=a.evidence_span,
        )
        db.add(item)
        db.flush()

        score = compute_risk(item, institution=extracted.institution)
        db.add(RiskScore(action_item_id=item.id, **score))
        saved.append(item)

        if score["score"] > highest_score:
            highest_score = score["score"]
        if a.deadline_iso is not None:
            if earliest_deadline is None or a.deadline_iso < earliest_deadline:
                earliest_deadline = a.deadline_iso

    letter.risk_score = highest_score
    letter.deadline_date = earliest_deadline

    # Attach the detected EUR amount to the most-severe action (max severity
    # rank, ties broken by highest risk score). Only one action carries it
    # so totals across letters don't double-count.
    if letter_amount is not None and saved:
        from app.models import Severity as _Sev

        sev_rank = {_Sev.CRITICAL: 4, _Sev.HIGH: 3, _Sev.MEDIUM: 2, _Sev.LOW: 1}
        amount_attached_to = max(saved, key=lambda x: sev_rank.get(x.severity, 0))
        amount_attached_to.amount_due_eur = letter_amount
        db.add(amount_attached_to)

    db.add(letter)
    return saved
