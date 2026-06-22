"""Action item CRUD + user-correction feedback loop (auth-required)."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.auth.dependencies import get_current_user
from app.database import get_session
from app.errors import ErrorCode, KlarHTTPException
from app.models import ActionItem, ActionStatus, Letter, User, UserCorrection
from app.schemas import ActionUpdate, ErrorResponse

router = APIRouter(prefix="/api/actions", tags=["actions"])


def _own_action(db: Session, action_id: UUID, user: User) -> ActionItem:
    item = db.get(ActionItem, action_id)
    if item is None:
        raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)
    letter = db.get(Letter, item.letter_id)
    if letter is None or letter.user_id != user.id:
        raise KlarHTTPException(404, ErrorCode.ACTION_NOT_FOUND)
    return item


@router.get(
    "",
    summary="List the current user's action items across all letters",
    description=(
        "Returns every action linked to a letter the user owns. Optional "
        "`?status=open|done|ignored` filter. Empty string is treated as "
        "'no filter' so the frontend can bind directly to React state."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        422: {"model": ErrorResponse, "description": "Unknown status value."},
    },
)
def list_actions(
    # `str | None` (not enum) so empty-string ?status= is treated as "no filter".
    status: str | None = Query(default=None),
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
        {
            "id": str(a.id),
            "letter_id": str(a.letter_id),
            "title": a.title,
            "deadline": a.deadline.isoformat() if a.deadline else None,
            "severity": a.severity.value,
            "status": a.status.value,
            "reply_needed": a.reply_needed,
        }
        for a in items
    ]


@router.patch(
    "/{action_id}",
    summary="Update fields on an action (status, deadline, title, description)",
    description=(
        "Every field change is logged to the `UserCorrection` table — the "
        "feedback loop for future prompt tuning. Send only the fields you "
        "want to change."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
        404: {"model": ErrorResponse, "description": "`ACTION_NOT_FOUND`."},
    },
)
def update_action(
    action_id: UUID,
    payload: ActionUpdate,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    item = _own_action(db, action_id, user)

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
    return {"id": str(item.id), "status": item.status.value}
