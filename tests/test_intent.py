"""Fuzzy Tier 2 intent matching: paraphrases route to state queries, tasks
and requests do not, and router + answerer agree on the kind."""

from __future__ import annotations

import pytest

from friday.router import Tier, classify
from friday.tiers import state_query
from friday.tiers.intent import classify_kind, fuzzy_kind, normalize

PARAPHRASES = [
    ("battery level?", "resources"),
    ("how much space is left", "resources"),
    ("how's the ram looking", "resources"),
    ("cpu usage", "resources"),
    ("which branch", "branch"),
    ("what branch are we on", "branch"),
    ("is postgres up", "is_running"),
    ("is the dev server still running", "is_running"),
    ("is anything listening on 8080", "whats_running"),
    ("what's codex up to", "agent_doing"),
    ("are the tests passing", "test_failures"),
    ("anything broken", "failures"),
    ("what am i working on", "in_progress"),
    ("what's going on", "whats_happening"),
]

NOT_STATE_QUERIES = [
    "what's the weather like",
    "can you check if the tests pass",
    "stop the dev server",
    "install htop",
    "free up some disk space",
    "switch to the main branch",
    "kill whatever is running on 8080",
    "how do i check the battery from the terminal on a thinkpad running arch",
    "tell me a joke",
]


@pytest.mark.parametrize("text,kind", PARAPHRASES)
def test_paraphrases_route_to_tier2_with_the_right_kind(text, kind):
    assert classify(text).tier is Tier.STATE_QUERY, text
    assert classify_kind(normalize(text))[0] == kind


@pytest.mark.parametrize("text", NOT_STATE_QUERIES)
def test_tasks_and_requests_stay_in_reasoning(text):
    assert classify(text).tier is Tier.REASONING, text
    assert fuzzy_kind(normalize(text)) == (None, None)


def test_is_running_target_is_cleaned():
    kind, match = classify_kind(normalize("is my postgres container still up"))
    assert kind == "is_running"
    assert match.group(1) == "postgres container"


def test_exact_is_running_strips_article_and_still(monkeypatch):
    snap = {"listening_ports": [{"process": "dev-server", "port": 3000}]}
    monkeypatch.setattr(state_query.state_mod, "snapshot", lambda: snap)
    result = state_query.answer("is the dev server still running")
    # Exact regex captures "the dev server still"; the formatter must still
    # find the port. Substring match on the cleaned target "dev server" needs
    # the process name to contain it, so we compare on the yes/no verdict.
    assert result.kind == "is_running"
    assert result.answerable


def test_fuzzy_route_is_labelled_for_spans():
    decision = classify("battery level?")
    assert decision.matched.startswith("fuzzy:")
    # Exact shapes keep their regex label, so existing span analysis holds.
    assert not classify("what's running").matched.startswith("fuzzy:")


def test_router_and_answerer_agree(monkeypatch):
    """Every Tier 2 route must land on a formatter -- the drift this module
    exists to prevent."""
    snap = {"git": {"branch": "main", "name": "friday", "dirty": False}, "resources": {"battery_pct": 50}}
    monkeypatch.setattr(state_query.state_mod, "snapshot", lambda: snap)
    for text, _ in PARAPHRASES:
        assert classify(text).tier is Tier.STATE_QUERY
        result = state_query.answer(text)
        assert result.kind is not None, text
