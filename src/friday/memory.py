"""Long-term memory: persisted turns/events, retrieved as a small relevance-
ranked slice, never dumped whole into a prompt.

One table, one `kind` column (working/episodic/semantic/procedural) rather
than four tables or four classes -- no query today needs the split. Rows
carry `project` and `created_at` so retrieval can filter by relevance,
project, recency, and importance, and so project-timeline questions
("what happened on X this week") are a plain WHERE clause.

Storage is stdlib `sqlite3` with an FTS5 virtual table (rung 3, no vector
DB, no embeddings, no ORM). FTS5 is compiled into every CPython build this
project targets, but availability is checked at connection time and raised
loudly rather than assumed -- see `_check_fts5`.

Consolidation replaces a set of raw related rows with one summary row: the
raw rows are marked `superseded_by = <summary id>` and excluded from
retrieval, rather than deleted, so history is not destroyed.

Satisfies `friday.brain.MemoryRetriever`: `async def retrieve(query) -> str`.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path.home() / ".friday" / "memory.db"

# Retrieval payload cap. ~4 chars/token is the accepted estimate for this
# project; 800 tokens * 4 chars/token.
MAX_RETRIEVE_CHARS = 800 * 4

KINDS = ("working", "episodic", "semantic", "procedural")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    superseded_by INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memories_project_created
    ON memories(project, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _check_fts5(conn: sqlite3.Connection) -> None:
    """Fail loudly (not silently degrade) if FTS5 isn't compiled in."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE __fts5_probe")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "sqlite3 build lacks FTS5 -- friday.memory requires it"
        ) from exc


# Characters FTS5's query-string parser treats specially. A stored memory may
# legitimately contain any of these; a *query* built from raw user speech
# must not let them reach MATCH unescaped, or it can throw (bad NEAR/column
# syntax) or silently change what matches.
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(raw: str) -> Optional[str]:
    """Turn free-text user input into a safe FTS5 MATCH query: every token
    double-quoted (which disables FTS5 operator syntax for it) and ANDed.
    Returns None if there are no indexable tokens."""
    tokens = _FTS_TOKEN_RE.findall(raw)
    if not tokens:
        return None
    return " AND ".join(f'"{t}"' for t in tokens)


class Memory:
    """FRIDAY's long-term memory store. Satisfies `brain.MemoryRetriever`."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        _check_fts5(self._conn)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes --------------------------------------------------------

    def record(
        self,
        text: str,
        *,
        kind: str = "episodic",
        project: str = "",
        importance: float = 0.0,
        created_at: Optional[float] = None,
    ) -> int:
        """Persist one raw event/turn. Returns its row id."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
        ts = time.time() if created_at is None else created_at
        cur = self._conn.execute(
            "INSERT INTO memories (kind, project, text, importance, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, project, text, importance, ts),
        )
        self._conn.commit()
        return cur.lastrowid

    def consolidate(
        self,
        raw_ids: list[int],
        summary: str,
        *,
        project: str = "",
        importance: float = 0.0,
        created_at: Optional[float] = None,
    ) -> int:
        """Replace `raw_ids` with one semantic summary row. The raw rows are
        marked `superseded_by` (not deleted) and excluded from retrieval."""
        summary_id = self.record(
            summary,
            kind="semantic",
            project=project,
            importance=importance,
            created_at=created_at,
        )
        self._conn.executemany(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            [(summary_id, rid) for rid in raw_ids],
        )
        self._conn.commit()
        return summary_id

    # -- reads -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        project: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 5,
    ) -> list[sqlite3.Row]:
        """Relevance-ranked (BM25) rows, filtered by project/recency,
        excluding superseded rows. `limit` caps the row count returned --
        callers needing every row in a window (timeline queries) pass a
        query of '' to skip the MATCH clause instead of raising `limit`."""
        clauses = ["m.superseded_by IS NULL"]
        params: list = []
        fts_q = _fts_query(query) if query else None

        if fts_q is not None:
            base = (
                "SELECT m.* FROM memories m "
                "JOIN memories_fts f ON f.rowid = m.id "
                "WHERE f.text MATCH ? AND "
            )
            params.append(fts_q)
            order = "bm25(memories_fts)"
        else:
            base = "SELECT m.* FROM memories m WHERE "
            order = "m.created_at DESC"

        if project:
            clauses.append("m.project = ?")
            params.append(project)
        if since is not None:
            clauses.append("m.created_at >= ?")
            params.append(since)

        sql = base + " AND ".join(clauses) + f" ORDER BY {order} LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    async def retrieve(self, query: str) -> str:
        """`MemoryRetriever` seam: top relevance-ranked rows for `query`,
        capped so it can never dump memory into the prompt. Runs on the
        stdlib sqlite3 connection, which is synchronous but fast enough
        (see tests) not to need a thread hop for this row count."""
        rows = self.search(query, limit=5)
        if not rows:
            return ""
        lines = []
        budget = MAX_RETRIEVE_CHARS
        for row in rows:
            line = f"[{row['project'] or 'general'}] {row['text']}"
            if len(line) + 1 > budget:
                remaining = budget - 1
                if remaining <= 0:
                    break
                lines.append(line[:remaining])
                break
            lines.append(line)
            budget -= len(line) + 1
        return "\n".join(lines)

    def timeline(self, project: str, *, since: float, limit: int = 200) -> list[sqlite3.Row]:
        """Every non-superseded row for `project` at or after `since`,
        newest first -- the project-timeline query ("what happened this
        week"), independent of text relevance."""
        return self.search("", project=project, since=since, limit=limit)
