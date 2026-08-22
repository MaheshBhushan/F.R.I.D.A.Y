"""Tests for friday.tiers.state_query (T7): the sub-100ms state-query tier.

Covers:
  1. end-to-end p99 < 100ms across >=20 turns, split cold-cache vs
     warm-cache, with zero llm_sent spans anywhere.
  2. answer correctness against live state for >=5 question kinds.
  3. honest degradation (+ escalation signal) for unanswerable-but-
     STATE_QUERY-shaped questions.
  4. wake.py's start_turn persistence fix actually writes a span record.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess

import pytest

from friday import state as state_mod
from friday.core.spans import start_turn
from friday.tiers import state_query
from friday.voice import wake as wake_mod

QUESTION_TEXTS = [
    "what's running",
    "what's happening",
    "what branch am I on",
    "what's in progress",
    "is sshd running",
    "is postgres running",
    "what's codex doing",
    "how's the battery",
    "what's the load",
]

UNANSWERABLE_TEXTS = [
    "any failures",
    "any errors",
    "how many tests failed",
]


def _pct(values_ns: list[int], q: float) -> float:
    """Percentile in ms from a list of nanosecond offsets."""
    if len(values_ns) == 1:
        return values_ns[0] / 1e6
    quantiles = statistics.quantiles(sorted(values_ns), n=100, method="inclusive")
    return quantiles[q - 1] / 1e6


_turn_counter = [0]


def _run_turn(tmp_path, text: str) -> dict:
    """Run one state-query turn through the real span machinery and
    return the persisted record. Each turn gets its own spans file so
    concurrent/looped calls in one test don't need to slice a shared log."""
    _turn_counter[0] += 1
    spans_path = tmp_path / f"spans-{_turn_counter[0]}.jsonl"
    with start_turn("state_query", path=spans_path) as span:
        result = state_query.answer(text, span=span)
    records = [json.loads(l) for l in spans_path.read_text().splitlines()]
    assert len(records) == 1
    return records[0], result


# --- Criterion 1 & 5: end-to-end latency, cold vs warm, no LLM spans --------


def test_e2e_latency_cold_and_warm_no_llm_spans(tmp_path):
    cold_offsets = []
    warm_offsets = []
    all_records = []

    texts = (QUESTION_TEXTS * 3)[:24]  # >= 20 turns, spread of kinds

    for i, text in enumerate(texts):
        # Force a cold cache on every other turn by resetting T5's TTL cache.
        cold = i % 2 == 0
        if cold:
            state_mod._cache = None
            state_mod._cache_time = 0.0
        else:
            state_mod.snapshot()  # warm it just before the turn

        record, result = _run_turn(tmp_path, text)
        all_records.append(record)
        offset = record["stages"]["task_complete"]
        (cold_offsets if cold else warm_offsets).append(offset)

    assert len(cold_offsets) >= 10
    assert len(warm_offsets) >= 10

    cold_p50, cold_p90, cold_p99 = (_pct(cold_offsets, q) for q in (50, 90, 99))
    warm_p50, warm_p90, warm_p99 = (_pct(warm_offsets, q) for q in (50, 90, 99))

    print("\nstate_query end-to-end latency (ms):")
    print(f"{'':<6}{'p50':>10}{'p90':>10}{'p99':>10}")
    print(f"{'cold':<6}{cold_p50:>10.3f}{cold_p90:>10.3f}{cold_p99:>10.3f}")
    print(f"{'warm':<6}{warm_p50:>10.3f}{warm_p90:>10.3f}{warm_p99:>10.3f}")

    for record in all_records:
        assert "llm_sent" not in record["stages"], "STATE_QUERY turn must never touch the LLM"

    # The cold number is the one that must meet the bar - it's the real
    # worst case since it pays the full snapshot() subprocess cost.
    assert cold_p99 < 100.0, f"cold p99 {cold_p99:.3f}ms >= 100ms"
    assert warm_p99 < 100.0, f"warm p99 {warm_p99:.3f}ms >= 100ms"


# --- Criterion 2: correctness against live state ----------------------------


def test_branch_answer_matches_git(tmp_path):
    actual = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    _, result = _run_turn(tmp_path, "what branch am I on")
    assert result.answerable
    assert actual in result.text, f"expected branch {actual!r} in answer {result.text!r}"


def test_dirty_state_matches_git_status(tmp_path):
    dirty_out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    actually_dirty = bool(dirty_out.strip())
    _, result = _run_turn(tmp_path, "what branch am I on")
    assert ("(dirty)" in result.text) == actually_dirty


def test_is_running_matches_ss_for_a_real_listener(tmp_path):
    ss_out = subprocess.run(["ss", "-tln"], capture_output=True, text=True, check=True).stdout
    # Find any port actually listening so the query is grounded in truth.
    lines = [l for l in ss_out.splitlines()[1:] if l.strip()]
    assert lines, "expected at least one listening socket on this machine to verify against"
    snap = state_mod.snapshot()
    ports = snap.get("listening_ports") or []
    named = [p for p in ports if p["process"]]
    assert named, "expected ss -tlnp to resolve at least one process name"
    target = named[0]["process"]
    _, result = _run_turn(tmp_path, f"is {target} running")
    assert result.answerable
    assert "Yes" in result.text
    assert target in result.text


def test_whats_running_process_counts_match_notable_processes(tmp_path):
    snap = state_mod.snapshot()
    expected_names = {p["name"] for p in (snap.get("notable_processes") or [])}
    _, result = _run_turn(tmp_path, "what's running")
    assert result.answerable
    if expected_names:
        # Every notable process name should be reflected somewhere in the
        # collapsed answer (as "N <name> session(s)" or "N <name> process(es)").
        for name in expected_names:
            assert name in result.text.lower() or name.capitalize() in result.text


def test_resources_answer_matches_proc_loadavg(tmp_path):
    with open("/proc/loadavg") as f:
        actual_load1 = float(f.read().split()[0])
    _, result = _run_turn(tmp_path, "what's the load")
    assert result.answerable
    assert f"{actual_load1:.1f}" in result.text


# --- Criterion 3: honest degradation + escalation signal --------------------


@pytest.mark.parametrize("text", UNANSWERABLE_TEXTS)
def test_unanswerable_state_queries_escalate_honestly(tmp_path, text):
    _, result = _run_turn(tmp_path, text)
    assert result.answerable is False
    assert result.escalate is True
    assert "don't have" in result.text.lower()
    # No fabricated content: must not claim a pass/fail count it never computed.
    assert "passed" not in result.text.lower()
    assert not any(ch.isdigit() for ch in result.text)


def test_unrecognized_state_query_shape_also_escalates(tmp_path):
    _, result = _run_turn(tmp_path, "what's the weather like")
    assert result.answerable is False
    assert result.escalate is True
    assert result.kind is None


# --- Criterion 4: wake.py persistence fix -----------------------------------


def test_listen_main_persists_span_via_with_form(tmp_path, monkeypatch):
    """Drive _listen_main's real `with start_turn(...)` code path using
    file-fed audio (test_data/alexa_test.wav) instead of a live mic, and
    confirm the span record actually lands on disk - proving the fix to
    the dropped-persistence bug."""
    spans_path = tmp_path / "spans.jsonl"
    chunks = wake_mod._iter_wav_chunks(wake_mod.TEST_WAV)

    async def fake_capture_loop(detector, on_detection, span=None):
        for chunk in chunks:
            detection = detector.feed_chunk(chunk, span=span)
            if detection is not None:
                on_detection(detection)
                return  # simulate the file-fed run ending after one detection

    monkeypatch.setattr(wake_mod, "capture_loop", fake_capture_loop)
    monkeypatch.setattr(
        wake_mod, "start_turn", lambda turn_kind: start_turn(turn_kind, path=spans_path)
    )

    asyncio.run(wake_mod._listen_main())

    assert spans_path.exists(), "no spans file written - persistence bug still present"
    records = [json.loads(l) for l in spans_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["turn_kind"] == "reflex"
    assert "speech_started" in records[0]["stages"]
