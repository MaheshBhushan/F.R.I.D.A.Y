"""Tests for the read-only Outlook tools: registration, token refresh, Graph
query shape, formatting, argument bounds and the not-signed-in path. Graph
and the Microsoft identity platform are replaced by a fake transport."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import stat
import time
from datetime import UTC, datetime

import pytest

from friday.permissions import Risk, risk_of
from friday.tools import TAINT_SOURCES, TOOL_SPECS, TOOLS, execute, outlook


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class FakeGraph:
    def __init__(self, payload=None, tokens=()):
        self.payload = payload or {"value": []}
        self.tokens = list(tokens)
        self.gets: list[dict] = []
        self.posts: list[dict] = []

    async def get(self, url, *, headers, params):
        self.gets.append({"url": url, "headers": headers, "params": params})
        return self.payload

    async def post_form(self, url, *, data):
        self.posts.append({"url": url, "data": data})
        return self.tokens.pop(0)


@pytest.fixture
def token_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIDAY_GRAPH_CLIENT_ID", "client-123")
    path = tmp_path / "graph_token.json"
    path.write_text(json.dumps({"access_token": "live", "refresh_token": "r1", "expires_at": time.time() + 600}))
    return path


def test_registered_read_only_and_mail_taints():
    for name in ("read_outlook_calendar", "read_outlook_mail"):
        assert name in TOOLS
        assert name in {s["name"] for s in TOOL_SPECS}
        assert risk_of(name) is Risk.READ_ONLY
    assert "read_outlook_mail" in TAINT_SOURCES
    assert "read_outlook_calendar" not in TAINT_SOURCES


def test_scopes_are_read_only():
    assert set(outlook.SCOPES.split()) == {"offline_access", "Calendars.Read", "Mail.Read"}


@sync
async def test_calendar_queries_window_and_formats(token_file):
    graph = FakeGraph({"value": [
        {"subject": "Standup", "start": {"dateTime": "2026-10-29T09:00:00.0000000"},
         "end": {"dateTime": "2026-10-29T09:15:00.0000000"}, "location": {"displayName": "Teams"},
         "organizer": {"emailAddress": {"name": "Ana"}}},
        {"subject": "Holiday", "isAllDay": True, "start": {"dateTime": "2026-10-30T00:00:00"},
         "end": {"dateTime": "2026-10-31T00:00:00"}},
        {"subject": "Dropped", "isCancelled": True, "start": {"dateTime": "2026-10-29T10:00:00"},
         "end": {"dateTime": "2026-10-29T11:00:00"}},
    ]})
    now = datetime(2026, 10, 29, 8, 0, tzinfo=UTC)
    out = await outlook.read_outlook_calendar(2, transport=graph, token_file=token_file, now=now)
    call = graph.gets[0]
    assert call["url"].endswith("/me/calendarView")
    assert call["headers"]["authorization"] == "Bearer live"
    assert call["params"]["startDateTime"] == "2026-10-29T08:00:00+00:00"
    assert call["params"]["endDateTime"] == "2026-10-31T08:00:00+00:00"
    assert "- 2026-10-29 09:00 to 09:15 UTC: Standup @ Teams (organizer Ana)" in out
    assert "all day 2026-10-30: Holiday" in out
    assert "Dropped" not in out
    assert graph.posts == []


@sync
async def test_mail_selects_previews_never_bodies(token_file):
    graph = FakeGraph({"value": [{
        "from": {"emailAddress": {"name": "Recruiter", "address": "r@example.test"}},
        "subject": "Interview\nslot", "receivedDateTime": "2026-10-29T07:30:00Z",
        "bodyPreview": "Hi,\n\nare you free " + "x" * 400, "isRead": False}]})
    out = await outlook.read_outlook_mail(3, True, transport=graph, token_file=token_file)
    params = graph.gets[0]["params"]
    assert "body" not in params["$select"].split(",")
    assert params["$filter"] == "isRead eq false" and params["$top"] == "3"
    assert "from Recruiter [unread]: Interview slot" in out
    assert "\n  Hi, are you free x" in out
    assert len(out.splitlines()[1]) <= outlook.MAX_PREVIEW_CHARS + 10


@sync
async def test_expired_token_refreshes_and_is_stored_private(token_file):
    token_file.write_text(json.dumps({"access_token": "old", "refresh_token": "r1", "expires_at": 0}))
    graph = FakeGraph(tokens=[{"access_token": "new", "refresh_token": "r2", "expires_in": 3600}])
    await outlook.read_outlook_mail(transport=graph, token_file=token_file)
    assert graph.posts[0]["data"]["grant_type"] == "refresh_token"
    assert graph.posts[0]["data"]["refresh_token"] == "r1"
    assert graph.gets[0]["headers"]["authorization"] == "Bearer new"
    assert json.loads(token_file.read_text())["refresh_token"] == "r2"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600


@sync
async def test_rejected_refresh_asks_for_login(token_file):
    token_file.write_text(json.dumps({"access_token": "old", "refresh_token": "r1", "expires_at": 0}))
    graph = FakeGraph(tokens=[{"error": "invalid_grant"}])
    with pytest.raises(RuntimeError, match="login"):
        await outlook.read_outlook_calendar(transport=graph, token_file=token_file)


@sync
async def test_device_login_polls_until_authorized(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIDAY_GRAPH_CLIENT_ID", "client-123")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(outlook.asyncio, "sleep", lambda _s: real_sleep(0))
    path = tmp_path / "t.json"
    graph = FakeGraph(tokens=[
        {"device_code": "dc", "message": "Go to https://microsoft.com/devicelogin", "interval": 1, "expires_in": 60},
        {"error": "authorization_pending"},
        {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
    ])
    out = io.StringIO()
    await outlook.login(graph, path, out=out)
    assert "devicelogin" in out.getvalue()
    assert graph.posts[0]["data"]["scope"] == outlook.SCOPES
    assert json.loads(path.read_text())["refresh_token"] == "r"


@sync
async def test_bounds_and_not_signed_in_are_tool_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(outlook, "TOKEN_FILE", tmp_path / "missing.json")
    outcome = await execute("t1", "read_outlook_mail", {"count": 50})
    assert outcome.is_error and "count must be" in outcome.content
    with pytest.raises(RuntimeError, match="not signed in"):
        await outlook.access_token(FakeGraph(), tmp_path / "missing.json")
