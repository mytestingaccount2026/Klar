"""FastAPI dependency that resolves the current user from the session cookie."""

import logging

from fastapi import Depends, Request, status
from sqlmodel import Session as DBSession, select

from app.config import settings
from app.database import get_session
from app.errors import ErrorCode, KlarHTTPException
from app.models import Session, User, utcnow

logger = logging.getLogger("klar.auth")


def _resolve_user(token: str | None, db: DBSession) -> User:
    if not token:
        raise KlarHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCode.AUTH_NOT_AUTHENTICATED,
            headers={"WWW-Authenticate": "Cookie"},
        )

    # Log the prefix of the token so the dev can correlate Set-Cookie ↔ lookup.
    # Full token never logged (it's a credential).
    token_prefix = token[:8]

    stmt = select(Session).where(Session.token == token)
    session_row = db.scalars(stmt).first()

    if session_row is None:
        # The cookie was sent but no Session row matches. Two real causes:
        #  - the backend was restarted with a different DATABASE_URL
        #  - the DB file was deleted between login and this request
        #  - the cookie was issued by a different uvicorn worker / process
        logger.warning(
            "AUTH_SESSION_NOT_FOUND: no Session row for token=%s... "
            "(DB=%s; possible causes: db wiped, multiple workers, env mismatch)",
            token_prefix,
            settings.database_url,
        )
        raise KlarHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCode.AUTH_SESSION_NOT_FOUND,
        )

    if session_row.expires_at < utcnow():
        logger.info(
            "AUTH_SESSION_EXPIRED: token=%s... expired_at=%s now=%s",
            token_prefix,
            session_row.expires_at,
            utcnow(),
        )
        raise KlarHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCode.AUTH_SESSION_EXPIRED,
        )

    user = db.get(User, session_row.user_id)
    if user is None:
        # Session row exists but the user it points to is gone — corrupt FK.
        logger.error(
            "AUTH: orphan Session row token=%s... user_id=%s has no User",
            token_prefix,
            session_row.user_id,
        )
        raise KlarHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCode.AUTH_SESSION_NOT_FOUND,
        )

    # Sliding-update last-seen (does NOT extend expiry).
    session_row.last_seen_at = utcnow()
    db.add(session_row)
    db.commit()
    return user


def get_current_user(request: Request, db: DBSession = Depends(get_session)) -> User:
    """Resolve the user from the session cookie. Raises 401 if unauthenticated."""
    token = request.cookies.get(settings.cookie_name)
    return _resolve_user(token, db)


def get_current_user_optional(
    request: Request,
    db: DBSession = Depends(get_session),
) -> User | None:
    """Same as get_current_user but returns None instead of raising 401."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    try:
        return _resolve_user(token, db)
    except KlarHTTPException:
        return None
