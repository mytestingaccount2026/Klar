"""GET /api/deadlines — view over ActionItem in the spec's deadline shape.

The spec's separate `deadlines` table is subsumed by ActionItem (per PRD §7,
which carries deadline_confidence + deadline_source + evidence_span on every
action). This router projects ActionItem rows into the spec's flat response
shape so the frontend's deadline calendar can render without knowing about
ActionItem.
"""

from datetime import date as _date

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.auth.dependencies import get_current_user
from app.database import get_session
from app.errors import ErrorCode, KlarHTTPException
from app.models import ActionItem, ActionStatus, Letter, RiskScore, User
from app.schemas import DeadlineItem, ErrorResponse

router = APIRouter(prefix="/api/deadlines", tags=["deadlines"])


@router.get(
    "",
    response_model=list[DeadlineItem],
    summary="List every deadline across the user's letters, sorted by due_date",
    description=(
        "A view over `ActionItem` projected into the spec's flat deadline "
        "shape (`id`, `letter_id`, `title`, `due_date`, `status`, "
        "`risk_score`, `severity`, `category`). Use this for the deadline "
        "calendar / urgency-sorted dashboard.\n\n"
        "By default, actions without a deadline are excluded. "
        "Pass `?include_no_date=true` to see all open actions regardless of "
        "whether a date was extracted."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        422: {"model": ErrorResponse, "description": "Unknown status value."},
    },
)
def list_deadlines(
    status: str | None = Query(default=None),
    include_no_date: bool = Query(default=False),
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Every action with a deadline, sorted by due_date ASC.

    By default, actions without a deadline are excluded. Pass
    `include_no_date=true` to also see actions without a date set.
    """
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
        select(ActionItem, Letter)
        .join(Letter, Letter.id == ActionItem.letter_id)
        .where(Letter.user_id == user.id)
    )
    if parsed_status:
        stmt = stmt.where(ActionItem.status == parsed_status)
    if not include_no_date:
        stmt = stmt.where(ActionItem.deadline.is_not(None))
    stmt = stmt.order_by(ActionItem.deadline.asc().nulls_last())

    # db.execute(stmt).all() returns Row tuples (ActionItem, Letter).
    # Using scalars() here would drop the Letter column — the join would still
    # apply at the SQL level but we'd lose category in Python and need N+1
    # re-fetches. With execute().all() we get the Letter row alongside.
    rows = db.execute(stmt).all()

    # Batch-load the latest risk score per ActionItem to avoid N+1 queries.
    action_ids = [a.id for a, _ in rows]
    risk_by_action: dict = {}
    if action_ids:
        risk_stmt = (
            select(RiskScore)
            .where(RiskScore.action_item_id.in_(action_ids))
            .order_by(RiskScore.computed_at.desc())
        )
        for rs in db.scalars(risk_stmt).all():
            # First (= most recent) wins because of the ORDER BY.
            risk_by_action.setdefault(rs.action_item_id, rs.score)

    out: list[DeadlineItem] = []
    for action, letter in rows:
        risk_value = risk_by_action.get(action.id, letter.risk_score)
        due = action.deadline or _date(9999, 12, 31)
        out.append(
            DeadlineItem(
                id=action.id,
                letter_id=action.letter_id,
                title=action.title,
                due_date=due,
                status=action.status,
                risk_score=risk_value,
                severity=action.severity,
                category=letter.category,
            )
        )
    return out
