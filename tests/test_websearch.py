"""Tests for the Exa-backed web_search tool: registration, offline execution
against a fake transport, query scrubbing, and the missing-key path."""

from __future__ import annotations

import asyncio
import functools

import pytest

from friday.permissions import PermissionDenied, Risk, risk_of
from friday.tools import RISK, TOOL_SPECS, TOOLS
from friday.tools.sanitize import safe_query
from friday.tools.websearch import MAX_OUTPUT_CHARS, MAX_RESULTS, web_search


def sync(fn):
    """Run an async test body on a fresh loop -- same plain-pytest style the
    rest of this suite uses (no pytest-asyncio dependency)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class FakeTransport:
    """Records the call and returns a canned Exa-shaped response."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, *, headers: dict, json: dict) -> dict:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.payload


def _result(i: int, text_len: int = 50) -> dict:
    return {
        "title": f"Result {i}",
        "url": f"https://example.com/{i}",
        "id": str(i),
        "text": "x" * text_len,
    }


# --- 1. registration ---------------------------------------------------------


def test_registered_in_tools_and_specs():
    assert "web_search" in TOOLS
    names = {spec["name"] for spec in TOOL_SPECS}
    assert "web_search" in names
    spec = next(s for s in TOOL_SPECS if s["name"] == "web_search")
    assert spec["input_schema"]["required"] == ["query"]
    assert "query" in spec["input_schema"]["properties"]


def test_risk_tier_is_read_only():
    assert risk_of("web_search") == Risk.READ_ONLY


def test_every_declared_tool_has_a_risk_tier():
    # The project's own invariant: no tool exists in one table without the other.
    assert set(TOOLS) == set(RISK)


# --- 2. works against the fake transport, output capped ---------------------


@sync
async def test_execute_returns_compact_ranked_list(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    payload = {"requestId": "abc", "results": [_result(1), _result(2)]}
    fake = FakeTransport(payload)

    out = await web_search("python asyncio queue backpressure", transport=fake)

    print("RESULT:\n", out)
    print("LENGTH:", len(out))
    assert "Result 1" in out
    assert "https://example.com/1" in out
    assert len(fake.calls) == 1
    assert fake.calls[0]["headers"]["x-api-key"] == "test-key"
    assert fake.calls[0]["json"]["query"] == "python asyncio queue backpressure"


@sync
async def test_output_is_hard_capped(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    # 50 long results, each with a 5000-char text blob -- way past any cap.
    payload = {"results": [_result(i, text_len=5000) for i in range(50)]}
    fake = FakeTransport(payload)

    out = await web_search("tmux capture-pane scrollback", num_results=50, transport=fake)

    print("CAPPED LENGTH:", len(out), "CAP:", MAX_OUTPUT_CHARS)
    # num_results is itself capped at MAX_RESULTS before hitting the transport.
    assert fake.calls[0]["json"]["numResults"] == MAX_RESULTS
    # and the formatted string itself never exceeds the hard cap (+ clip notice).
    assert len(out) <= MAX_OUTPUT_CHARS + 60


# --- 3. query scrubbing -------------------------------------------------------


REFUSED_QUERIES = [
    ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789", "sk-ant- key prefix"),
    ("sk-abcdefghijklmnopqrstuvwxyzABCDEF", "sk- key prefix"),
    ("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "ghp_ prefix"),
    ("github_pat_abcdefghijklmnopqrstuvwxyz0123456789", "github_pat_ prefix"),
    ("xoxb-1234567890-abcdefghijklmnop", "xoxb- prefix"),
    ("AKIAABCDEFGHIJKLMNOP", "AKIA prefix"),
    ("AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz012", "AIza prefix"),
    ("glpat-abcdefghijklmnopqrst", "glpat- prefix"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIB...", "PEM block"),
    ("aB3xQ9mK7pL2vN8wR5tY1zC4dF6gH0jU1234", "40-char high-entropy token"),
    ("please check my .env file for the answer", ".env filename"),
    ("what is inside id_rsa", "id_rsa filename"),
    ("look at server.pem for the cert", ".pem filename"),
    ("known_hosts entries seem wrong", "known_hosts filename"),
    ("/home/maheshk/.ssh/id_rsa", "absolute path under /home"),
    ("/proc/self/environ", "absolute path under /proc"),
    ("x" * 4001, "oversized query"),
]


@pytest.mark.parametrize("query,label", REFUSED_QUERIES)
def test_safe_query_refuses(query, label):
    with pytest.raises(PermissionDenied):
        safe_query(query)
    print(f"REFUSED ({label}): {query[:60]!r}...")


LEGITIMATE_QUERIES = [
    "postgres connection refused fix",
    "python asyncio queue backpressure",
    "tmux capture-pane scrollback",
]


@pytest.mark.parametrize("query", LEGITIMATE_QUERIES)
def test_safe_query_allows_legitimate_searches(query):
    result = safe_query(query)
    print(f"ALLOWED: {query!r} -> {result!r}")
    assert result == query


@sync
async def test_scrubbing_happens_before_any_transport_call(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    fake = FakeTransport({"results": []})
    with pytest.raises(PermissionDenied):
        await web_search("ghp_abcdefghijklmnopqrstuvwxyz0123456789", transport=fake)
    assert fake.calls == []  # never reached the network seam


# --- 4. missing key -----------------------------------------------------------


@sync
async def test_missing_api_key_fails_clearly_without_a_request(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    fake = FakeTransport({"results": []})

    with pytest.raises(RuntimeError) as exc_info:
        await web_search("postgres connection refused fix", transport=fake)

    print("ERROR:", exc_info.value)
    assert "EXA_API_KEY" in str(exc_info.value)
    assert fake.calls == []


# --- 5. no network in tests ----------------------------------------------------
# Every test above passes `transport=` explicitly, so `HttpxTransport` (the
# only code path in this module that touches the network) is never
# constructed anywhere in this file -- confirmed by inspection: no test omits
# `transport=`, and grepping this file for `HttpxTransport` finds zero uses.
