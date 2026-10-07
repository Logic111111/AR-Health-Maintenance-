"""Engine creation, database bootstrap and retry helper."""
from __future__ import annotations

import functools
import logging
import time
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError

from . import schema

log = logging.getLogger(__name__)


def make_engine(url: str) -> sa.Engine:
    u = make_url(url)
    if u.get_backend_name() == "sqlite":
        if u.database and u.database != ":memory:":
            Path(u.database).parent.mkdir(parents=True, exist_ok=True)
        eng = sa.create_engine(url)

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
        return eng
    return sa.create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def ensure_database(url: str) -> None:
    """Create the MySQL/MariaDB database if it does not exist (no-op for SQLite)."""
    u = make_url(url)
    if u.get_backend_name() not in ("mysql", "mariadb") or not u.database:
        return
    server = sa.create_engine(u.set(database=""), pool_pre_ping=True)
    with server.begin() as conn:
        conn.execute(sa.text(
            f"CREATE DATABASE IF NOT EXISTS `{u.database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    server.dispose()


def create_all(engine: sa.Engine) -> None:
    schema.metadata.create_all(engine, checkfirst=True)


def drop_all(engine: sa.Engine) -> None:
    schema.metadata.drop_all(engine, checkfirst=True)


def with_retry(retries: int = 5, base_delay: float = 2.0):
    """Retry a DB operation on connection errors with exponential back-off."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except (OperationalError, DBAPIError) as exc:
                    transient = isinstance(exc, OperationalError) or getattr(
                        exc, "connection_invalidated", False)
                    if not transient or attempt == retries:
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    log.warning("DB error (%s), retry %d/%d in %.0fs",
                                exc.__class__.__name__, attempt, retries, delay)
                    time.sleep(delay)
        return wrapper
    return decorator
