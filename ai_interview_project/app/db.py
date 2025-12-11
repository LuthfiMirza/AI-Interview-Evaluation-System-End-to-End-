"""Database session and base class configuration."""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
print("🔥🔥 DEBUG: DATABASE_URL loaded from env =", DATABASE_URL)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")


url = make_url(DATABASE_URL)
LOGGER = logging.getLogger(__name__)
schema = url.query.get("schema")
if schema:
    query = dict(url.query)
    query.pop("schema", None)
    url = url.set(query=query or None)

engine = create_engine(str(url), pool_pre_ping=True)

if schema:

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_connection, _) -> None:
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}"')

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


def init_db() -> None:
    """Create database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    LOGGER.info("Database schema ensured on %s", url.render_as_string(hide_password=True))


@contextmanager
def session_scope() -> Iterator[sessionmaker]:
    """Provide a transactional scope around database operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        raise
    finally:
        session.close()
