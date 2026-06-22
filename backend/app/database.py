"""SQLite + SQLModel engine and session helpers."""

from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings


def _connect_args() -> dict:
    if settings.database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(settings.database_url, connect_args=_connect_args())


def init_db() -> None:
    if settings.database_url.startswith("sqlite:///"):
        path = settings.database_url.replace("sqlite:///", "", 1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    from app import models  # noqa: F401  — populate SQLModel metadata

    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
