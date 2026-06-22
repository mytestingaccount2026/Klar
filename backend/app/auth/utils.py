"""Password hashing + opaque session/reset token helpers."""

import secrets
from datetime import datetime, timedelta

from app.models import utcnow

import bcrypt

from app.config import settings


def hash_password(plain: str) -> str:
    """bcrypt with cost factor 12 — ~250ms on a modern laptop."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode(
        "utf-8"
    )


def verify_password(plain: str, password_hash: str) -> bool:
    if not password_hash:
        # User has no password set (e.g., legacy row) — never authenticate.
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# Precomputed valid bcrypt hash used as a timing-attack defense in /login.
# Computed once at module import so unknown-email branches still pay the full
# bcrypt cost (preventing user enumeration via response timing).
DUMMY_PASSWORD_HASH: str = bcrypt.hashpw(
    b"klar-dummy-never-matches-anything", bcrypt.gensalt(rounds=12)
).decode("utf-8")


def generate_session_token() -> str:
    """256 bits of cryptographic randomness, URL-safe base64."""
    return secrets.token_urlsafe(32)


def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)


def session_expiry() -> datetime:
    return utcnow() + timedelta(hours=settings.session_ttl_hours)


def reset_token_expiry() -> datetime:
    return utcnow() + timedelta(minutes=settings.reset_token_ttl_minutes)
