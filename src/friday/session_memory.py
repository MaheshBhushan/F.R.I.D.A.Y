"""Persistent hot conversation memory: completed user/assistant turn pairs."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from friday.core import events
from friday.memory import DEFAULT_DB_PATH

DEFAULT_TURNS = 10
MAX_TURNS = 50
DEFAULT_CONTEXT_CHARS = 12_000
MAX_CONTEXT_CHARS = 200_000


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class ConversationMessage:
    id: Optional[int]
    session_id: str
    turn_id: str
    role: str
    content: str
    created_at: float


@dataclass(frozen=True)
class ConversationTurn:
    session_id: str
    turn_id: str
    user: ConversationMessage
    assistant: ConversationMessage
    route_tier: str = ""
    interrupted: bool = False
    metadata: Optional[dict[str, Any]] = None


_MIGRATIONS = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at REAL NOT NULL,
            last_activity_at REAL NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversation_turns (
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            status TEXT NOT NULL,
            route_tier TEXT NOT NULL DEFAULT '',
            interrupted INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            completed_at REAL,
            failure_reason TEXT,
            PRIMARY KEY (session_id, turn_id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (session_id, turn_id)
                REFERENCES conversation_turns(session_id, turn_id)
        );
        CREATE TABLE IF NOT EXISTS memory_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
            ON conversation_messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_turn
            ON conversation_messages(session_id, turn_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_turns_completed
            ON conversation_turns(session_id, status, completed_at);
        """,
    ),
)


class SessionMemory:
    """SQLite persistence plus an in-memory cache of recent completed turns."""

    def __init__(
        self,
        path: Path = DEFAULT_DB_PATH,
        *,
        turn_limit: Optional[int] = None,
        context_chars: Optional[int] = None,
        clock=time.time,
    ) -> None:
        self.path = Path(path)
        self.turn_limit = (
            _env_int("FRIDAY_SESSION_MEMORY_TURNS", DEFAULT_TURNS, 0, MAX_TURNS)
            if turn_limit is None else turn_limit
        )
        self.context_chars = (
            _env_int(
                "FRIDAY_SESSION_MEMORY_CONTEXT_CHARS",
                DEFAULT_CONTEXT_CHARS,
                0,
                MAX_CONTEXT_CHARS,
            )
            if context_chars is None else context_chars
        )
        if not 0 <= self.turn_limit <= MAX_TURNS:
            raise ValueError(f"turn_limit must be between 0 and {MAX_TURNS}")
        if not 0 <= self.context_chars <= MAX_CONTEXT_CHARS:
            raise ValueError(f"context_chars must be between 0 and {MAX_CONTEXT_CHARS}")
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cache: deque[ConversationTurn] = deque(maxlen=self.turn_limit)
        self._pending: dict[str, ConversationMessage] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._errors = 0
        self.session_id = uuid.uuid4().hex
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=2.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 2000")
            self._migrate()
            self.session_id = self._restore_session()
            self._hydrate()
        except Exception as exc:  # noqa: BLE001 - memory must degrade, not kill voice
            self._disable("init", exc)

    def _migrate(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        applied = {
            row[0]
            for row in self._conn.execute(
                "SELECT version FROM session_schema_migrations"
            ).fetchall()
        }
        for version, sql in _MIGRATIONS:
            if version in applied:
                continue
            with self._conn:
                self._conn.executescript(sql)
                self._conn.execute(
                    "INSERT INTO session_schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, self._clock()),
                )

    def _restore_session(self) -> str:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value FROM memory_state WHERE key = 'active_session_id'"
        ).fetchone()
        if row is not None:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (row[0],)
            ).fetchone()
            if exists is not None:
                return str(row[0])
        session_id = uuid.uuid4().hex
        now = self._clock()
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions(id, started_at, last_activity_at, status) "
                "VALUES (?, ?, ?, 'active')",
                (session_id, now, now),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_state(key, value) VALUES "
                "('active_session_id', ?)",
                (session_id,),
            )
        return session_id

    def _hydrate(self) -> None:
        if self._conn is None or self.turn_limit == 0:
            return
        rows = self._conn.execute(
            "SELECT * FROM conversation_turns WHERE session_id = ? AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT ?",
            (self.session_id, self.turn_limit),
        ).fetchall()
        for row in reversed(rows):
            turn = self._load_turn(row)
            if turn is not None:
                self._cache.append(turn)

    def _load_turn(self, row: sqlite3.Row) -> Optional[ConversationTurn]:
        assert self._conn is not None
        messages = self._conn.execute(
            "SELECT * FROM conversation_messages WHERE session_id = ? AND turn_id = ? "
            "ORDER BY id",
            (row["session_id"], row["turn_id"]),
        ).fetchall()
        by_role = {message["role"]: message for message in messages}
        if "user" not in by_role or "assistant" not in by_role:
            return None

        def message(role: str) -> ConversationMessage:
            item = by_role[role]
            return ConversationMessage(
                item["id"], item["session_id"], item["turn_id"], item["role"],
                item["content"], item["created_at"],
            )

        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return ConversationTurn(
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            user=message("user"),
            assistant=message("assistant"),
            route_tier=row["route_tier"],
            interrupted=bool(row["interrupted"]),
            metadata=metadata,
        )

    def _disable(self, operation: str, exc: Exception) -> None:
        self._errors += 1
        events.emit(
            "memory-error", operation, count=self._errors,
            error=events.quote(repr(exc)),
        )
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _report_unavailable(self, operation: str) -> None:
        if self._errors:
            self._errors += 1
            events.emit(
                "memory-error", operation, count=self._errors,
                error="persistence unavailable; hot cache only",
            )

    async def start(self) -> None:
        events.emit(
            "memory-init", session=self.session_id,
            restored_turns=len(self._cache), persistent=self._conn is not None,
        )

    async def recent_turns(self, limit: Optional[int] = None) -> list[ConversationTurn]:
        async with self._lock:
            count = self.turn_limit if limit is None else max(0, min(limit, self.turn_limit))
            return list(self._cache)[-count:] if count else []

    async def begin_turn(
        self,
        user_text: str,
        *,
        turn_id: Optional[str] = None,
        route_tier: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        text = user_text.strip()
        if not text:
            return ""
        turn_id = turn_id or uuid.uuid4().hex
        now = self._clock()
        user = ConversationMessage(None, self.session_id, turn_id, "user", text, now)
        async with self._lock:
            self._pending[turn_id] = user
            if self._conn is not None:
                try:
                    with self._conn:
                        self._conn.execute(
                            "INSERT INTO conversation_turns "
                            "(session_id, turn_id, status, route_tier, metadata_json, created_at) "
                            "VALUES (?, ?, 'pending', ?, ?, ?)",
                            (self.session_id, turn_id, route_tier,
                             json.dumps(metadata or {}, sort_keys=True), now),
                        )
                        self._conn.execute(
                            "INSERT INTO conversation_messages "
                            "(session_id, turn_id, role, content, created_at) "
                            "VALUES (?, ?, 'user', ?, ?)",
                            (self.session_id, turn_id, text, now),
                        )
                        self._conn.execute(
                            "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                            (now, self.session_id),
                        )
                except Exception as exc:  # noqa: BLE001
                    self._disable("begin", exc)
            else:
                self._report_unavailable("begin")
        events.debug("memory-turn", turn=turn_id, status="pending", route=route_tier)
        return turn_id

    async def complete_turn(
        self,
        turn_id: str,
        assistant_text: str,
        *,
        route_tier: str = "",
        interrupted: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        text = assistant_text.strip()
        if not turn_id or not text:
            return False
        started = time.perf_counter()
        async with self._lock:
            user = self._pending.get(turn_id)
            if user is None and self._conn is not None:
                try:
                    row = self._conn.execute(
                        "SELECT * FROM conversation_messages WHERE session_id = ? "
                        "AND turn_id = ? AND role = 'user' ORDER BY id LIMIT 1",
                        (self.session_id, turn_id),
                    ).fetchone()
                    if row is not None:
                        user = ConversationMessage(
                            row["id"], row["session_id"], row["turn_id"], row["role"],
                            row["content"], row["created_at"],
                        )
                except Exception as exc:  # noqa: BLE001
                    self._disable("read-pending", exc)
            if user is None:
                return False
            now = self._clock()
            assistant = ConversationMessage(
                None, self.session_id, turn_id, "assistant", text, now
            )
            if self._conn is not None:
                try:
                    with self._conn:
                        cur = self._conn.execute(
                            "INSERT INTO conversation_messages "
                            "(session_id, turn_id, role, content, created_at) "
                            "VALUES (?, ?, 'assistant', ?, ?)",
                            (self.session_id, turn_id, text, now),
                        )
                        assistant = ConversationMessage(
                            cur.lastrowid, self.session_id, turn_id, "assistant", text, now
                        )
                        self._conn.execute(
                            "UPDATE conversation_turns SET status = 'completed', route_tier = ?, "
                            "interrupted = ?, metadata_json = ?, completed_at = ? "
                            "WHERE session_id = ? AND turn_id = ?",
                            (route_tier, int(interrupted),
                             json.dumps(metadata or {}, sort_keys=True), now,
                             self.session_id, turn_id),
                        )
                        self._conn.execute(
                            "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                            (now, self.session_id),
                        )
                except Exception as exc:  # noqa: BLE001
                    self._disable("complete", exc)
            else:
                self._report_unavailable("complete")
            self._pending.pop(turn_id, None)
            self._cache.append(
                ConversationTurn(
                    self.session_id, turn_id, user, assistant,
                    route_tier, interrupted, metadata or {},
                )
            )
        events.emit(
            "memory-turn", turn=turn_id, status="completed", route=route_tier,
            interrupted=interrupted,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return True

    async def fail_turn(self, turn_id: str, reason: str) -> None:
        if not turn_id:
            return
        async with self._lock:
            self._pending.pop(turn_id, None)
            if self._conn is not None:
                try:
                    with self._conn:
                        self._conn.execute(
                            "UPDATE conversation_turns SET status = 'failed', failure_reason = ? "
                            "WHERE session_id = ? AND turn_id = ?",
                            (reason[:500], self.session_id, turn_id),
                        )
                except Exception as exc:  # noqa: BLE001
                    self._disable("fail", exc)
            else:
                self._report_unavailable("fail")
        events.emit("memory-turn", turn=turn_id, status="failed")

    async def context_messages(self, limit: Optional[int] = None) -> list[dict[str, str]]:
        started = time.perf_counter()
        turns = await self.recent_turns(limit)
        selected: list[ConversationTurn] = []
        used = 0
        for turn in reversed(turns):
            size = len(turn.user.content) + len(turn.assistant.content)
            if used + size > self.context_chars:
                break
            selected.append(turn)
            used += size
        selected.reverse()
        messages: list[dict[str, str]] = []
        for turn in selected:
            messages.append({"role": "user", "content": turn.user.content})
            assistant = turn.assistant.content
            if turn.interrupted:
                assistant += "\n[Response was interrupted before playback completed.]"
            messages.append({"role": "assistant", "content": assistant})
        events.emit(
            "memory-context", turns=len(selected), messages=len(messages), chars=used,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return messages

    async def new_session(self) -> str:
        async with self._lock:
            old = self.session_id
            new = uuid.uuid4().hex
            now = self._clock()
            if self._conn is not None:
                try:
                    with self._conn:
                        self._conn.execute(
                            "UPDATE sessions SET status = 'closed', last_activity_at = ? WHERE id = ?",
                            (now, old),
                        )
                        self._conn.execute(
                            "INSERT INTO sessions(id, started_at, last_activity_at, status) "
                            "VALUES (?, ?, ?, 'active')",
                            (new, now, now),
                        )
                        self._conn.execute(
                            "INSERT OR REPLACE INTO memory_state(key, value) VALUES "
                            "('active_session_id', ?)",
                            (new,),
                        )
                except Exception as exc:  # noqa: BLE001
                    self._disable("new-session", exc)
            else:
                self._report_unavailable("new-session")
            self.session_id = new
            self._cache.clear()
            self._pending.clear()
        events.emit("memory-session", frm=old, to=new)
        return new

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as exc:  # noqa: BLE001
                    self._disable("close", exc)
                finally:
                    self._conn = None
