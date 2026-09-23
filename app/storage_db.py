from __future__ import annotations
import sqlite3
import json
import os
from typing import Optional
from .models import Session

DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "pace_pal.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at REAL,
            data TEXT
        )
        """
    )
    return conn


def save_session(session: Session) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO sessions (id, created_at, data) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data, created_at = excluded.created_at",
            (session.id, session.createdAt, session.model_dump_json()),
        )
    conn.close()


def get_session(session_id: str) -> Optional[Session]:
    conn = _connect()
    row = conn.execute("SELECT data FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return Session.model_validate(json.loads(row[0]))


def list_sessions() -> list[Session]:
    conn = _connect()
    rows = conn.execute("SELECT data FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [Session.model_validate(json.loads(r[0])) for r in rows]
