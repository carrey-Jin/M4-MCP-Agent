"""
Database connection helpers. Set DATABASE_URL in .env, or rely on default SQLite file.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_DEFAULT_SQLITE = BASE_DIR / "app.db"


def m4_connect_database(database_url: str | None = None) -> Engine:
    """
    Create and return a SQLAlchemy Engine (pool). Credentials must come from DATABASE_URL
    in .env or from the database_url argument — never hardcode passwords in code.

    When database_url is omitted, reads DATABASE_URL from the environment.
    If unset, falls back to a local SQLite file at ./app.db.

    Examples for DATABASE_URL:
        sqlite:///./app.db
        postgresql+psycopg://user:pass@localhost:5432/dbname
        mysql+pymysql://user:pass@localhost:3306/dbname
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        url = f"sqlite:///{_DEFAULT_SQLITE}"

    # SQLite needs check_same_thread=False when used across threads (e.g. web workers).
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(url, pool_pre_ping=True, connect_args=connect_args or None)


@contextmanager
def m4_database_connection(
    database_url: str | None = None,
) -> Generator[Connection, None, None]:
    """
    Context manager yielding a single Connection; commits on success, rolls back on error.
    """
    engine = m4_connect_database(database_url)
    with engine.connect() as conn:
        with conn.begin():
            yield conn
