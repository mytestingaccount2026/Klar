"""Klar error envelope + machine-readable error codes.

Why this exists
===============
The frontend dev should never have to string-match on `detail`. Every error
the backend produces — whether raised explicitly, surfaced from Pydantic
validation, or bubbled up from an unhandled exception — comes back wrapped in
the same shape:

    {
      "code":    "AUTH_INVALID_CREDENTIALS",   // stable identifier
      "message": "Email or password is incorrect.",   // user-facing copy
      "details": {...} | null                  // optional structured extras
    }

For SSE error events the same shape lives inside the `data:` payload:

    event: error
    data: {"code": "EXTRACTION_FAILED", "message": "We couldn't read this letter."}

How to raise an error
=====================
    from app.errors import ErrorCode, KlarHTTPException

    raise KlarHTTPException(
        status_code=404,
        code=ErrorCode.LETTER_NOT_FOUND,
        message="That letter doesn't exist or you don't have access to it.",
    )

The exception handler in `app.main` does the rest — JSON-encodes the envelope
and sets the status code. Plain `HTTPException` still works (gets `code=
HTTP_ERROR`) so existing 3rd-party error paths don't crash, but new code
should use `KlarHTTPException`.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("klar.errors")


class ErrorCode(str, Enum):
    """Stable identifier for every error the API can produce.

    Frontend code should switch/case on `response.code` (NOT on `message`).
    The string values here are part of the public API contract — never rename
    without bumping a version.
    """

    # --- generic ---
    HTTP_ERROR = "HTTP_ERROR"  # untyped fallback (legacy HTTPException)
    INTERNAL_ERROR = "INTERNAL_ERROR"  # unhandled exception
    VALIDATION_ERROR = "VALIDATION_ERROR"  # request body / query / path

    # --- auth ---
    AUTH_NOT_AUTHENTICATED = "AUTH_NOT_AUTHENTICATED"  # no cookie at all
    AUTH_SESSION_NOT_FOUND = (
        "AUTH_SESSION_NOT_FOUND"  # cookie present, but no Session row in DB
    )
    AUTH_SESSION_EXPIRED = (
        "AUTH_SESSION_EXPIRED"  # Session row exists but past expires_at
    )
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"  # wrong email / password
    AUTH_EMAIL_TAKEN = "AUTH_EMAIL_TAKEN"  # signup with existing email
    AUTH_INVALID_RESET_TOKEN = (
        "AUTH_INVALID_RESET_TOKEN"  # token unknown / already used
    )
    AUTH_RESET_TOKEN_EXPIRED = "AUTH_RESET_TOKEN_EXPIRED"  # token past 15-min TTL

    # --- letters ---
    LETTER_NOT_FOUND = "LETTER_NOT_FOUND"
    LETTER_FILE_MISSING = "LETTER_FILE_MISSING"  # row exists but file gone
    LETTER_EMPTY_UPLOAD = "LETTER_EMPTY_UPLOAD"
    LETTER_TOO_LARGE = "LETTER_TOO_LARGE"
    LETTER_UNSUPPORTED_TYPE = "LETTER_UNSUPPORTED_TYPE"
    LETTER_CORRUPT_FILE = "LETTER_CORRUPT_FILE"  # magic-bytes mismatch
    LETTER_MIME_MISMATCH = "LETTER_MIME_MISMATCH"  # declared ≠ detected

    # --- actions ---
    ACTION_NOT_FOUND = "ACTION_NOT_FOUND"

    # --- pipeline / AI ---
    EXTRACTION_FAILED = (
        "EXTRACTION_FAILED"  # SSE-only: model returned no tool call, parse error, etc.
    )
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"  # network / 5xx from Qwen
    PDF_RENDER_FAILED = "PDF_RENDER_FAILED"  # pdf2image / poppler missing


# User-facing default messages per code. Keep short, no jargon, no secrets.
# Routes can override per call by passing `message=`.
_DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.HTTP_ERROR: "Something went wrong with that request.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on our end. Please try again.",
    ErrorCode.VALIDATION_ERROR: "Some fields in your request are invalid.",
    ErrorCode.AUTH_NOT_AUTHENTICATED: "Please sign in to continue.",
    ErrorCode.AUTH_SESSION_NOT_FOUND: "Your session is no longer recognized. Please sign in again.",
    ErrorCode.AUTH_SESSION_EXPIRED: "Your session has expired. Please sign in again.",
    ErrorCode.AUTH_INVALID_CREDENTIALS: "Email or password is incorrect.",
    ErrorCode.AUTH_EMAIL_TAKEN: "An account with that email already exists.",
    ErrorCode.AUTH_INVALID_RESET_TOKEN: "This reset link is invalid or has already been used.",
    ErrorCode.AUTH_RESET_TOKEN_EXPIRED: "This reset link has expired. Please request a new one.",
    ErrorCode.LETTER_NOT_FOUND: "That letter doesn't exist or you don't have access to it.",
    ErrorCode.LETTER_FILE_MISSING: "We can't find the uploaded file for this letter.",
    ErrorCode.LETTER_EMPTY_UPLOAD: "The uploaded file is empty.",
    ErrorCode.LETTER_TOO_LARGE: "That file is too large. Maximum is 10 MB.",
    ErrorCode.LETTER_UNSUPPORTED_TYPE: "We can only read JPEG, PNG, HEIC, WebP, or PDF letters.",
    ErrorCode.LETTER_CORRUPT_FILE: "The file looks corrupted or isn't the type it claims to be.",
    ErrorCode.LETTER_MIME_MISMATCH: "The file's content doesn't match its declared type.",
    ErrorCode.ACTION_NOT_FOUND: "That action doesn't exist or you don't have access to it.",
    ErrorCode.EXTRACTION_FAILED: "We couldn't read this letter. Try a clearer photo or PDF.",
    ErrorCode.LLM_PROVIDER_ERROR: "Our AI provider is having trouble. Please try again in a moment.",
    ErrorCode.PDF_RENDER_FAILED: "We couldn't open that PDF. Try uploading it as an image instead.",
}


class KlarHTTPException(HTTPException):
    """HTTPException carrying a machine-readable Klar `ErrorCode`.

    Use this everywhere we raise an HTTP error in our own code. The
    `code` is what the frontend switches on; the `message` is what the
    user sees. Both ship in the JSON envelope.
    """

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message or _DEFAULT_MESSAGES.get(code, "Something went wrong.")
        self.details = details
        # HTTPException's `detail` carries the JSON we ultimately render — we
        # keep the envelope shape consistent here too so frameworks that
        # render HTTPException directly (rare) still produce sensible output.
        super().__init__(
            status_code=status_code,
            detail={
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            },
            headers=headers,
        )


def _envelope(code: ErrorCode, message: str, details: Any = None) -> dict[str, Any]:
    """The error envelope every non-2xx response uses.

    The `detail` key is a wire-compatibility alias so clients that read the
    FastAPI default `{"detail": "..."}` shape (notably the
    `docs/06-frontend-integration-contract.md` clients) still get a usable
    string. Sophisticated clients should read `code` for branching.
    """
    return {
        "code": code.value,
        "message": message,
        "detail": message,  # wire-compat: clients reading FastAPI default shape
        "details": details,
    }


# ---------- exception handlers (registered in app.main) ----------


async def klar_exception_handler(_req: Request, exc: KlarHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message, exc.details),
        headers=exc.headers,
    )


async def generic_http_exception_handler(
    _req: Request, exc: HTTPException
) -> JSONResponse:
    """Catches plain `HTTPException`s (FastAPI's default, 3rd-party code)
    and reshapes them into the Klar envelope so the frontend sees a single
    error format everywhere."""
    # If detail is already a Klar envelope dict (from KlarHTTPException
    # routing through the default handler), pass it through.
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code, content=exc.detail, headers=exc.headers
        )

    # Auth-shaped status codes get more specific codes by default.
    code = ErrorCode.HTTP_ERROR
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        code = ErrorCode.AUTH_NOT_AUTHENTICATED

    message = str(exc.detail) if exc.detail else _DEFAULT_MESSAGES[code]
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, message),
        headers=exc.headers,
    )


async def validation_exception_handler(
    _req: Request, exc: RequestValidationError
) -> JSONResponse:
    """Translate Pydantic's verbose validation errors into the Klar envelope.

    Frontend gets `details.errors = [{field, message}, ...]` — a stable shape
    that's easy to bind to form-level error UI.
    """
    errors: list[dict[str, str]] = []
    for err in exc.errors():
        # Build a dotted path from the `loc` tuple: ('body', 'email') → 'email'
        # Drop the first element ('body', 'query', 'path') — frontend doesn't care.
        loc = list(err.get("loc", []))
        if loc and loc[0] in ("body", "query", "path", "header", "cookie"):
            loc = loc[1:]
        field = ".".join(str(p) for p in loc) or "(root)"
        errors.append({"field": field, "message": err.get("msg", "Invalid value")})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_envelope(
            ErrorCode.VALIDATION_ERROR,
            _DEFAULT_MESSAGES[ErrorCode.VALIDATION_ERROR],
            {"errors": errors},
        ),
    )


async def unhandled_exception_handler(req: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: never leaks an internal error or stack trace to the user.

    Logs the full traceback server-side, returns a generic 500 envelope.
    """
    logger.exception(
        "Unhandled exception on %s %s: %s",
        req.method,
        req.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope(
            ErrorCode.INTERNAL_ERROR,
            _DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR],
        ),
    )


# ---------- helpers for SSE error events ----------


def sse_error_payload(
    code: ErrorCode, message: str | None = None, details: Any = None
) -> dict[str, Any]:
    """Same envelope as HTTP responses — for use in SSE `error` events."""
    return _envelope(code, message or _DEFAULT_MESSAGES[code], details)
