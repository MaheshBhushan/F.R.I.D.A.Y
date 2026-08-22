"""Tests for friday.memory: seeded-corpus retrieval quality/latency, the
payload cap, the brain.MemoryRetriever seam, consolidation, project
timelines, and FTS5 injection safety.

No network, no real DB: every test opens `Memory` against a pytest tmp_path.
"""

from __future__ import annotations

import asyncio
import functools
import random
import sqlite3
import time

import pytest

from friday import brain
from friday.brain import FakeLLMTransport, ScriptedTurn, TurnResult, complete, ordered_blocks
from friday.memory import MAX_RETRIEVE_CHARS, Memory


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


PROJECTS = ["FinanRAG", "mk-fuzz", "agent-overlay", "vektordb", "mk-fleet"]
DAY = 86400.0


def _seed_corpus(mem: Memory, *, n: int = 200) -> dict:
    """Seed n synthetic turns across PROJECTS and dates. Returns markers:
    a handful of rows we know the exact expected retrieval answer for."""
    rng = random.Random(0)
    now = time.time()
    filler_verbs = ["debugged", "refactored", "profiled", "reviewed", "deployed"]
    filler_nouns = ["the parser", "the scheduler", "a flaky test", "the config loader"]

    for i in range(n):
        project = rng.choice(PROJECTS)
        days_ago = rng.uniform(0, 40)
        text = f"{rng.choice(filler_verbs)} {rng.choice(filler_nouns)} on {project}"
        mem.record(text, kind="episodic", project=project, created_at=now - days_ago * DAY)

    # Markers: distinctive text we can search for unambiguously.
    marker_id = mem.record(
        "FinanRAG: fixed the token-validation bug in the EDGAR fetch client, "
        "confirmed with pytest, all green",
        kind="episodic",
        project="FinanRAG",
        created_at=now - 1 * DAY,
    )
    recent_id = mem.record(
        "mk-fuzz: reran the LinUCB bandit benchmark against AFL havoc weights",
        kind="episodic",
        project="mk-fuzz",
        created_at=now - 2 * DAY,
    )
    old_id = mem.record(
        "mk-fuzz: initial scaffold for the learned scheduler experiment",
        kind="episodic",
        project="mk-fuzz",
        created_at=now - 20 * DAY,
    )
    return {"now": now, "marker_id": marker_id, "recent_id": recent_id, "old_id": old_id}


# --- 1. relevance + project + recency filter: correctness and latency ------


def test_relevance_project_recency_query_is_fast_and_correct(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    marks = _seed_corpus(mem)

    latencies = []
    rows = None
    for _ in range(25):
        t0 = time.perf_counter()
        rows = mem.search(
            "token-validation bug EDGAR",
            project="FinanRAG",
            since=marks["now"] - 7 * DAY,
            limit=5,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99) - 1]
    print(f"\n[criterion 1] p50={p50:.3f}ms p99={p99:.3f}ms over {len(latencies)} queries")

    assert p99 < 30.0, f"p99 {p99:.3f}ms exceeds 30ms budget"
    assert len(rows) >= 1
    assert rows[0]["id"] == marks["marker_id"], "top hit is not the actually-relevant row"
    assert all(r["project"] == "FinanRAG" for r in rows), "project filter leaked other projects"
    cutoff = marks["now"] - 7 * DAY
    assert all(r["created_at"] >= cutoff for r in rows), "recency filter let old rows through"
    mem.close()


# --- 2. retrieval payload is capped, even when everything matches ----------


@sync
async def test_retrieve_is_capped_even_when_everything_matches(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    # 200 rows that all match the query term "project".
    for i in range(200):
        mem.record(
            f"project update number {i}: did a bunch of project work on project things",
            kind="episodic",
            project="mk-fleet",
        )

    result = await mem.retrieve("project")
    char_len = len(result)
    est_tokens = char_len / 4
    print(f"\n[criterion 2] retrieve() length={char_len} chars, ~{est_tokens:.0f} tokens "
          f"(cap {MAX_RETRIEVE_CHARS} chars / 800 tokens)")

    assert char_len <= MAX_RETRIEVE_CHARS
    assert est_tokens <= 800
    mem.close()


def test_retrieve_empty_when_no_match(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    mem.record("something unrelated", project="x")

    async def run():
        return await mem.retrieve("zzz_no_such_token_zzz")

    assert asyncio.run(run()) == ""
    mem.close()


# --- 3. satisfies brain.MemoryRetriever, lands in volatile section ---------


@sync
async def test_memory_satisfies_brain_retriever_seam_in_volatile_section(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    mem.record(
        "FinanRAG: fixed the token-validation bug, tests green, committed",
        kind="episodic",
        project="FinanRAG",
    )

    from friday.brain import AssembledContext

    context = AssembledContext(
        world_state="Project: FinanRAG @ main",
        memory=await mem.retrieve("token-validation bug"),
        assembled_at=time.monotonic(),
        query="what happened with the token bug",
    )
    assert "token-validation" in context.memory

    transport = FakeLLMTransport([ScriptedTurn(tokens=("It's fixed.",))])
    result = TurnResult()
    tokens = []
    async for tok in complete(
        "did we fix the token bug", transport, context=context, result=result
    ):
        tokens.append(tok)

    request = transport.requests[0]
    blocks = ordered_blocks(request)
    print(f"\n[criterion 3] ordered_blocks={blocks}")

    # memory text must appear in the volatile user-content block, not in the
    # cached static system prefix.
    system_text = request["system"][0]["text"]
    assert "token-validation" not in system_text
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}

    user_blocks = request["messages"][-1]["content"]
    volatile_text = "\n".join(b["text"] for b in user_blocks if b["type"] == "text")
    assert "token-validation" in volatile_text
    assert blocks[-1] == "user:world_state" or "user:world_state" in blocks
    mem.close()


# --- 4. consolidation replaces raw rows with a summary ---------------------


@sync
async def test_consolidation_supersedes_raw_rows(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    r1 = mem.record("Codex opened token_validate.py and found the off-by-one", project="FinanRAG")
    r2 = mem.record("Codex patched the bounds check in token_validate.py", project="FinanRAG")
    r3 = mem.record("pytest run: 42 passed, 0 failed after the patch", project="FinanRAG")
    r4 = mem.record("git commit: fix token validation off-by-one", project="FinanRAG")

    before = await mem.retrieve("token validation")
    print(f"\n[criterion 4] before consolidation:\n{before}")
    assert "off-by-one" in before or "bounds check" in before

    summary_id = mem.consolidate(
        [r1, r2, r3, r4],
        "Codex found the token-validation bug, fixed it, tests green, committed.",
        project="FinanRAG",
        importance=0.8,
    )

    after = await mem.retrieve("token validation")
    print(f"[criterion 4] after consolidation:\n{after}")

    assert "Codex found the token-validation bug, fixed it, tests green, committed." in after
    for raw_id in (r1, r2, r3, r4):
        assert str(raw_id) not in after  # raw text shouldn't leak back in

    raw_rows = mem.search("off-by-one", project="FinanRAG", limit=10)
    assert raw_rows == [], "superseded raw rows still surfaced by search"

    row = mem._conn.execute(
        "SELECT superseded_by FROM memories WHERE id = ?", (r1,)
    ).fetchone()
    assert row["superseded_by"] == summary_id
    mem.close()


# --- 5. project-timeline query, recency genuinely excludes old rows --------


def test_project_timeline_excludes_rows_older_than_window(tmp_path):
    mem = Memory(tmp_path / "memory.db")
    now = time.time()

    recent_texts = [
        "FinanRAG: benchmarked flat FAISS vs Graph-RAG over real EDGAR filings",
        "FinanRAG: wrote T0 and T1 evaluation harness",
        "FinanRAG: found a retrieval regression, root-caused to chunk overlap",
    ]
    for i, text in enumerate(recent_texts):
        mem.record(text, project="FinanRAG", created_at=now - (i + 1) * DAY)

    old_texts = [
        "FinanRAG: initial repo scaffold",
        "FinanRAG: first pass at the EDGAR downloader",
    ]
    for i, text in enumerate(old_texts):
        mem.record(text, project="FinanRAG", created_at=now - (10 + i) * DAY)

    mem.record("mk-fuzz: unrelated project, should never appear", created_at=now - 1 * DAY)

    since = now - 7 * DAY
    rows = mem.timeline("FinanRAG", since=since)
    texts = [r["text"] for r in rows]
    print(f"\n[criterion 5] timeline (last 7 days) = {texts}")

    assert len(rows) == 3
    assert all("FinanRAG" == r["project"] for r in rows)
    assert all(r["created_at"] >= since for r in rows)
    assert not any("scaffold" in t or "first pass" in t for t in texts)
    mem.close()


# --- 6. injection safety: SQL / FTS5 metacharacters round-trip safely ------


def test_injection_safe_storage_and_query(tmp_path):
    mem = Memory(tmp_path / "memory.db")

    malicious_text = '\'; DROP TABLE memories; -- NEAR/2 "quoted" * chars'
    row_id = mem.record(malicious_text, project="security-test")

    # DB must still exist and be queryable -- nothing got dropped.
    row = mem._conn.execute(
        "SELECT text FROM memories WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["text"] == malicious_text

    tables = {
        r[0]
        for r in mem._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "memories" in tables

    # Now use the same metacharacters *as an untrusted query* -- this must
    # not raise (bad FTS5 syntax) and must not become a bypass search.
    malicious_query = '\'; DROP TABLE memories; -- NEAR/2 "quoted" * chars'
    rows = mem.search(malicious_query, project="security-test", limit=5)
    assert any(r["id"] == row_id for r in rows), "legit tokens in the query should still match"

    # Table still intact after querying with it.
    tables_after = {
        r[0]
        for r in mem._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert tables_after == tables

    # A query that's pure FTS5 operator syntax and no real tokens (e.g. only
    # symbols) must not raise either.
    rows2 = mem.search('" NEAR/4 * "', limit=5)
    assert isinstance(rows2, list)

    print("\n[criterion 6] malicious text/query round-tripped safely, no table dropped")
    mem.close()


def test_fts5_available_on_this_python():
    import sqlite3 as s

    conn = s.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.close()
