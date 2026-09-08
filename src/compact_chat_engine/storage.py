from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional

from .models import Message, SessionState


class SqliteSessionStorage:
    """
    Relational SQLite storage for session states and conversation history.
    Includes automated legacy JSON migration support.
    """

    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    scratchpad TEXT DEFAULT '',
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, is_archived)
                """
            )

    def auto_migrate_legacy_json(self, session_id: str, recent_buffer_count: int = 4) -> bool:
        """
        Detects legacy `session_{session_id}.json` and imports it into SQLite if not already stored.
        """
        legacy_file = f"session_{session_id}.json"
        if not os.path.exists(legacy_file):
            return False

        # If already exists in database, skip migration
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                return False

        try:
            with open(legacy_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scratchpad = ""
            recent_messages: list[Message] = []
            archive: list[Message] = []

            # Format 1: Legacy flat array
            if isinstance(data, list):
                turns = [m for m in data if m.get("role") != "system"]
                if len(turns) <= recent_buffer_count:
                    recent_messages = [Message.from_dict(m) for m in turns]
                else:
                    recent_messages = [
                        Message.from_dict(m) for m in turns[-recent_buffer_count:]
                    ]
                    archive = [
                        Message.from_dict(m) for m in turns[:-recent_buffer_count]
                    ]
            # Format 2: Dual-layer dict schema
            elif isinstance(data, dict):
                scratchpad = data.get("scratchpad", "")
                recent_messages = [
                    Message.from_dict(m) for m in data.get("recent_messages", [])
                ]
                archive = [Message.from_dict(m) for m in data.get("archive", [])]

            state = SessionState(
                session_id=session_id,
                scratchpad=scratchpad,
                recent_messages=recent_messages,
                archive=archive,
            )
            self.save_session(state)

            # Rename migrated file to avoid re-migration
            migrated_name = f"{legacy_file}.migrated"
            try:
                os.replace(legacy_file, migrated_name)
            except OSError:
                pass

            return True
        except Exception as e:
            print(f"[Warning] Failed to migrate legacy file '{legacy_file}': {e}")
            return False

    def load_session(self, session_id: str, recent_buffer_count: int = 4) -> SessionState:
        # Check for legacy JSON migration first
        self.auto_migrate_legacy_json(session_id, recent_buffer_count=recent_buffer_count)

        with self._get_connection() as conn:
            session_row = conn.execute(
                "SELECT session_id, scratchpad FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if not session_row:
                return SessionState(session_id=session_id)

            scratchpad = session_row["scratchpad"]

            message_rows = conn.execute(
                """
                SELECT role, content, is_archived, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

            recent: list[Message] = []
            archive: list[Message] = []
            for row in message_rows:
                msg = Message(
                    role=row["role"],
                    content=row["content"],
                    is_archived=bool(row["is_archived"]),
                    created_at=row["created_at"],
                )
                if msg.is_archived:
                    archive.append(msg)
                else:
                    recent.append(msg)

            return SessionState(
                session_id=session_id,
                scratchpad=scratchpad,
                recent_messages=recent,
                archive=archive,
            )

    def save_session(self, state: SessionState) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, scratchpad, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    scratchpad = excluded.scratchpad,
                    updated_at = excluded.updated_at
                """,
                (state.session_id, state.scratchpad, now),
            )

            conn.execute("DELETE FROM messages WHERE session_id = ?", (state.session_id,))

            params = []
            for msg in state.archive:
                params.append(
                    (state.session_id, msg.role, msg.content, 1, msg.created_at)
                )
            for msg in state.recent_messages:
                params.append(
                    (state.session_id, msg.role, msg.content, 0, msg.created_at)
                )

            if params:
                conn.executemany(
                    """
                    INSERT INTO messages (session_id, role, content, is_archived, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    params,
                )

    def list_sessions(self) -> list[str]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [row["session_id"] for row in rows]

    def list_sessions_detailed(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            query = """
                SELECT 
                    s.session_id,
                    s.scratchpad,
                    s.updated_at,
                    SUM(CASE WHEN m.is_archived = 0 THEN 1 ELSE 0 END) as active_count,
                    SUM(CASE WHEN m.is_archived = 1 THEN 1 ELSE 0 END) as archived_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """
            rows = conn.execute(query).fetchall()
            results: list[dict[str, Any]] = []
            for r in rows:
                scratchpad_val = r["scratchpad"] or ""
                results.append({
                    "session_id": r["session_id"],
                    "has_scratchpad": bool(scratchpad_val.strip()),
                    "updated_at": r["updated_at"],
                    "active_count": r["active_count"] or 0,
                    "archived_count": r["archived_count"] or 0,
                })
            return results

    def delete_session(self, session_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
