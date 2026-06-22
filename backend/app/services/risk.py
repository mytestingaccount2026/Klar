"""Deterministic risk score — formula from DeadlinePivot PRD §4.5.

The LLM only provides `severity` and `institution`. The score itself is computed
server-side via a weighted formula so the urgency number is reproducible and
defensible to users.
"""

from datetime import date

from app.models import ActionItem, Severity

INSTITUTION_WEIGHTS: dict[str, float] = {
    "Ausländerbehörde": 1.00,
    "Auslaenderbehoerde": 1.00,
    "AOK": 0.85,
    "TK": 0.85,
    "Krankenversicherung": 0.85,
    "Finanzamt": 0.90,
    "Universität": 0.80,
    "Universitaet": 0.80,
    "Studentenwerk": 0.75,
    "Bank": 0.70,
    "Sparkasse": 0.70,
    "Vermieter": 0.60,
    "Landlord": 0.60,
    "Beitragsservice": 0.55,
    "GEZ": 0.55,
}

SEVERITY_SCORE: dict[Severity, float] = {
    Severity.CRITICAL: 1.00,
    Severity.HIGH: 0.75,
    Severity.MEDIUM: 0.50,
    Severity.LOW: 0.25,
}


def _deadline_proximity(deadline: date | None) -> float:
    if deadline is None:
        return 0.50
    days_left = (deadline - date.today()).days
    if days_left < 0:
        return 1.00
    if days_left == 0:
        return 0.95
    if days_left <= 3:
        return 0.85
    if days_left <= 7:
        return 0.65
    if days_left <= 14:
        return 0.45
    if days_left <= 30:
        return 0.25
    return 0.10


def _institution_weight(institution: str) -> float:
    if not institution:
        return 0.40
    key = institution.strip()
    if key in INSTITUTION_WEIGHTS:
        return INSTITUTION_WEIGHTS[key]
    lower = key.lower()
    for known, weight in INSTITUTION_WEIGHTS.items():
        if known.lower() in lower:
            return weight
    return 0.40


def _missing_info_penalty(item: ActionItem) -> float:
    penalty = 0.0
    if item.deadline is None:
        penalty += 0.5
    if not item.evidence_span:
        penalty += 0.3
    if not item.steps:
        penalty += 0.2
    return min(penalty, 1.0)


def compute_risk(item: ActionItem, institution: str) -> dict:
    dp = _deadline_proximity(item.deadline)
    iw = _institution_weight(institution)
    sp = SEVERITY_SCORE.get(item.severity, 0.50)
    mp = _missing_info_penalty(item)

    raw = dp * 0.40 + iw * 0.30 + sp * 0.20 + mp * 0.10
    score = int(round(raw * 100))

    explanation = " · ".join(
        [
            f"deadline_proximity={dp:.2f} (×0.40)",
            f"institution_weight={iw:.2f} (×0.30)",
            f"severity={sp:.2f} (×0.20)",
            f"missing_info_penalty={mp:.2f} (×0.10)",
        ]
    )
    result = {
        "score": score,
        "deadline_proximity_pts": dp,
        "institution_weight": iw,
        "severity_pts": sp,
        "missing_info_penalty": mp,
        "explanation": explanation,
    }
    return result
