"""Engine and session management. SQLite default; any SQLAlchemy URL works."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from marketiq.config.settings import get_settings
from marketiq.data.models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _make_engine(url: str, echo: bool = False) -> Engine:
    kwargs = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        db_path = url.replace("sqlite:///", "")
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return engine


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        s = get_settings()
        _engine = _make_engine(s.database.resolved_url(), s.database.echo)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def configure_engine(url: str, echo: bool = False) -> Engine:
    """Point the application at a different database (used by tests and the CLI)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = _make_engine(url, echo)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(engine: Optional[Engine] = None) -> None:
    Base.metadata.create_all(engine or get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Plain session for FastAPI dependencies; caller closes."""
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()
