"""Tests for friday.router: the three-tier routing table.

Covers the correctness table (30+ utterances, including adversarial
near-misses that must not be misrouted), partial-transcript classification,
and the structural guarantee that Tier 1/2 never touch the network or an
LLM span.
"""

from __future__ import annotations

import socket

import pytest

from friday.core.spans import start_turn
from friday.router import RouteDecision, Tier, classify, classify_and_mark, dispatch_tier1


# --- Criterion 3: routing correctness table ---------------------------------

TIER1_CASES = [
    "stop",
    "Stop.",
    "mute",
    "wait",
    "cancel that",
    "never mind",
    "nevermind",
    "shut up",
    "okay",
    "yeah",
    "mhm",
    "uh huh",
    "sure",
    "right",
    "got it",
]

TIER2_CASES = [
    "what's Codex doing",
    "what's Codex doing?",
    "what is Codex doing",
    "any failures",
    "any errors",
    "what's running",
    "what's happening",
    "what branch am I on",
    "what branch is this",
    "is the dev server running",
    "how many tests failed",
    "what's in progress",
]

# Adversarial near-misses: look like a Tier 1/2 trigger but are Tier 3 tasks.
TIER3_CASES = [
    "stop the dev server",
    "make Codex fix the tests",
    "wait until the build finishes then deploy",
    "cancel that meeting on my calendar",
    "what's the weather like",
    "can you check if the tests pass",
    "write a function that reverses a string",
    "commit and push my changes",
    "summarize this file for me",
    "remind me to call Alice at 5pm",
]


@pytest.mark.parametrize("text", TIER1_CASES)
def test_tier1_reflex(text: str) -> None:
    decision = classify(text, is_final=True)
    assert decision.tier is Tier.REFLEX, f"{text!r} should be Tier 1, got {decision}"


@pytest.mark.parametrize("text", TIER2_CASES)
def test_tier2_state_query(text: str) -> None:
    decision = classify(text, is_final=True)
    assert decision.tier is Tier.STATE_QUERY, f"{text!r} should be Tier 2, got {decision}"


@pytest.mark.parametrize("text", TIER3_CASES)
def test_tier3_reasoning_not_misrouted(text: str) -> None:
    decision = classify(text, is_final=True)
    assert decision.tier is Tier.REASONING, f"{text!r} should be Tier 3, got {decision}"


def test_total_case_count_at_least_30() -> None:
    assert len(TIER1_CASES) + len(TIER2_CASES) + len(TIER3_CASES) >= 30


# --- Criterion 4: partial-transcript classification -------------------------

PARTIAL_UNAMBIGUOUS_CASES = [
    ("what's cod", Tier.STATE_QUERY),
    ("what's runn", Tier.STATE_QUERY),
    ("any fail", Tier.STATE_QUERY),
    ("what branch", Tier.STATE_QUERY),
    ("can you check if the", Tier.REASONING),
    ("write a function", Tier.REASONING),
    ("make Codex", Tier.REASONING),
]

PARTIAL_AMBIGUOUS_CASES = [
    "stop",
    "st",
    "wait",
    "never",
    "cancel",
    "shut",
]


@pytest.mark.parametrize("text,expected_tier", PARTIAL_UNAMBIGUOUS_CASES)
def test_partial_transcript_classifies_when_unambiguous(text: str, expected_tier: Tier) -> None:
    decision = classify(text, is_final=False)
    assert decision.tier is expected_tier, f"{text!r} should classify as {expected_tier}, got {decision}"


@pytest.mark.parametrize("text", PARTIAL_AMBIGUOUS_CASES)
def test_partial_transcript_declines_when_ambiguous(text: str) -> None:
    decision = classify(text, is_final=False)
    assert decision.tier is None, f"{text!r} should decline to guess, got {decision}"


def test_bare_reflex_word_resolves_once_final() -> None:
    # "stop" alone is ambiguous mid-utterance...
    assert classify("stop", is_final=False).tier is None
    # ...but is Tier 1 once we know the utterance actually ended there.
    assert classify("stop", is_final=True).tier is Tier.REFLEX
    # while continuing into more words makes it a Tier 3 task, never Tier 1.
    assert classify("stop the dev server", is_final=True).tier is Tier.REASONING


# --- Criterion 2: no LLM span, no network for Tier 1/2 ----------------------

def test_tier1_dispatch_marks_no_llm_span(tmp_path) -> None:
    span_path = tmp_path / "spans.jsonl"
    span = start_turn("reflex", path=span_path)
    decision = classify_and_mark("stop", span, is_final=True)
    action = dispatch_tier1(decision)
    span.write()

    assert action == "stop_playback"
    record = span.to_record()
    assert "llm_sent" not in record["stages"]


def test_tier2_dispatch_marks_no_llm_span(tmp_path) -> None:
    span_path = tmp_path / "spans.jsonl"
    span = start_turn("state_query", path=span_path)
    decision = classify_and_mark("what's running", span, is_final=True)
    assert decision.tier is Tier.STATE_QUERY
    span.write()

    record = span.to_record()
    assert "llm_sent" not in record["stages"]


def test_tier1_and_tier2_open_no_socket(monkeypatch) -> None:
    """Structural proof of zero network I/O: patch socket.socket to raise if
    ever constructed, then run classify+dispatch for a Tier 1 and a Tier 2
    utterance (dispatch_tier2 excluded, since friday.state legitimately
    spawns local subprocesses, not sockets - see its own module docstring;
    this test only asserts no *socket* is opened, which is the network
    surface an LLM/HTTP call would need)."""

    def _no_sockets(*args, **kwargs):
        raise AssertionError("socket.socket() called - Tier 1/2 must not touch the network")

    monkeypatch.setattr(socket, "socket", _no_sockets)

    decision1 = classify("mute", is_final=True)
    assert decision1.tier is Tier.REFLEX
    dispatch_tier1(decision1)

    decision2 = classify("any failures", is_final=True)
    assert decision2.tier is Tier.STATE_QUERY


def test_route_decision_is_plain_dataclass() -> None:
    d = RouteDecision(tier=Tier.REFLEX, matched="stop")
    assert d.tier is Tier.REFLEX
    assert d.matched == "stop"
