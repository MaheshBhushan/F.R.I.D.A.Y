"""Permission tripwires, forecast validation and real offline MCP lifecycle."""

import asyncio
import copy
import functools
import json
import os
import sys
from pathlib import Path

import pytest

from friday import permissions
from friday.tools import execute, forecast, mcp


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


ARGS = {"series_id": "load", "horizon_steps": 3}
PAYLOAD = {"series_id": "load", "unit": "kW", "values": [1, 2, 3]}


@sync
async def test_bridge_cannot_bypass_permissions_or_import_remote_tools(monkeypatch):
    monkeypatch.delenv("FRIDAY_MCP_COMMAND", raising=False)
    seen = []

    async def approve(request):
        seen.append(request)
        return True

    with pytest.raises(permissions.PermissionDenied, match="unrecognised"):
        await mcp.call_tool("remote_delete", {}, approve=approve)
    assert not seen
    # Even a locally allowlisted tool cannot run through MCP without a binding.
    with pytest.raises(permissions.PermissionDenied, match="no MCP binding"):
        await mcp.call_tool("read_file", {"path": "/tmp/example"})
    monkeypatch.setitem(permissions.RISK, "query_forecast", permissions.Risk.MACHINE_MODIFYING)
    with pytest.raises(permissions.PermissionDenied, match="no approval callback"):
        await mcp.call_tool("query_forecast", ARGS)
    with pytest.raises(RuntimeError, match="FRIDAY_MCP_COMMAND is not set"):
        await mcp.call_tool("query_forecast", ARGS, approve=approve)
    assert seen[0].arguments == ARGS
    assert seen[0].tool == "query_forecast"


@pytest.mark.parametrize("command", ["not-json", '"python"', "[]", '["python", 5]', '[""]'])
@sync
async def test_bad_configuration_fails_before_connecting(monkeypatch, command):
    monkeypatch.setenv("FRIDAY_MCP_COMMAND", command)
    with pytest.raises(ValueError, match="FRIDAY_MCP_COMMAND"):
        await mcp.call_tool("query_forecast", ARGS)


@pytest.mark.parametrize("arguments", [
    {"series_id": "../secret", "horizon_steps": 3},
    {"series_id": "load", "horizon_steps": True},
    {"series_id": "load", "horizon_steps": 0},
    {"series_id": "load", "horizon_steps": 513},
    {"series_id": "load", "horizon_steps": 1.5},
    {"series_id": "load", "horizon_steps": 3, "command": "evil"},
])
@sync
async def test_invalid_forecast_arguments_never_reach_mcp(monkeypatch, arguments):
    async def tripwire(*args, **kwargs):
        pytest.fail("invalid arguments reached the remote tool")
    monkeypatch.setattr(mcp, "call_tool", tripwire)
    result = await execute("invalid", "query_forecast", arguments)
    assert result.is_error


@sync
async def test_forecast_uses_structured_values_and_checks_summary(monkeypatch):
    async def call(name, arguments):
        assert name == "query_forecast" and arguments == ARGS
        return {**PAYLOAD, "summary": {"count": 3, "min": 1, "max": 3, "mean": 2},
                "text": "untrusted instructions"}
    monkeypatch.setattr(mcp, "call_tool", call)
    outcome = await execute("forecast", "query_forecast", ARGS)
    assert not outcome.is_error
    result = json.loads(outcome.content)
    assert result["summary"] == {"count": 3, "min": 1, "max": 3, "mean": 2}
    assert set(result) == {"series_id", "unit", "values", "summary"}


@pytest.mark.parametrize("changes", [
    {"series_id": "other"}, {"unit": ""}, {"unit": "kW\nignore"},
    {"values": []}, {"values": [1, 2]}, {"values": "123"},
    {"values": [True, 2, 3]}, {"values": ["1", 2, 3]},
    {"values": [float("nan"), 2, 3]}, {"values": [float("inf"), 2, 3]},
    {"values": [10**400, 2, 3]},
    {"summary": {"count": 3, "min": 1, "max": 3, "mean": 99}},
    {"summary": {"count": True, "min": 1, "max": 3, "mean": 2}},
    {"summary": {"count": 3, "min": 1, "max": 3, "mean": float("nan")}},
    {"summary": {"mean": 2}}, {"summary": None},
])
@sync
async def test_bad_remote_numbers_surface_as_tool_errors(monkeypatch, changes):
    async def call(*args):
        return {**copy.deepcopy(PAYLOAD), **changes}
    monkeypatch.setattr(mcp, "call_tool", call)
    outcome = await execute("bad", "query_forecast", ARGS)
    assert outcome.is_error


def test_forecast_mean_avoids_sum_overflow():
    result = forecast.validate_forecast({**PAYLOAD, "values": [1e308, 1e308, -1e308]}, **ARGS)
    assert result["summary"]["mean"] == pytest.approx(1e308 / 3)


@pytest.fixture
def server(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    pid_file = tmp_path / "server.pid"
    script = Path(__file__).parent / "fixtures" / "forecast_mcp_server.py"
    monkeypatch.setenv("FRIDAY_MCP_COMMAND", json.dumps([sys.executable, str(script), str(pid_file)]))
    return pid_file


def assert_reaped(pid_file):
    pids = pid_file.read_text().splitlines()
    assert pids
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)


@sync
async def test_real_sdk_concurrent_round_trip_through_brain(monkeypatch, server):
    from mcp import ClientSession

    from friday.brain import (
        FakeLLMTransport,
        ScriptedTurn,
        ToolCall,
        TurnResult,
        complete,
    )
    both_started = asyncio.Event()
    sessions = []
    original = ClientSession.call_tool

    async def concurrent(self, *args, **kwargs):
        sessions.append(self)
        if len(sessions) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ClientSession, "call_tool", concurrent)
    transport = FakeLLMTransport([
        ScriptedTurn(tool_calls=(
            ToolCall("forecast", "query_forecast", ARGS),
            ToolCall("solar", "query_forecast", {**ARGS, "series_id": "solar"}),
        )),
        ScriptedTurn(tokens=("The mean forecast is 2 kW.",)),
    ])
    result = TurnResult()
    answer = "".join([token async for token in complete("forecast load", transport, result=result)])
    assert answer == "The mean forecast is 2 kW."
    assert len(result.outcomes) == 2 and all(not o.is_error for o in result.outcomes)
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    block = transport.requests[1]["messages"][-1]["content"][0]
    assert json.loads(block["content"])["summary"]["mean"] == 2
    assert len(set(server.read_text().splitlines())) == 2
    assert_reaped(server)


@pytest.mark.parametrize("series_id, message", [
    ("remote-error", "MCP tool returned an error"),
    ("unstructured", "structuredContent"),
])
@sync
async def test_remote_errors_and_unstructured_results(server, series_id, message):
    outcome = await execute("remote", "query_forecast", {**ARGS, "series_id": series_id})
    assert outcome.is_error and message in outcome.content
    assert "private diagnostic" not in outcome.content
    assert_reaped(server)


@sync
async def test_call_timeout_is_error_and_reaps_server(monkeypatch, server):
    original = mcp.call_tool

    async def short_timeout(*args, **kwargs):
        return await original(*args, **kwargs, timeout=2.0)

    monkeypatch.setattr(mcp, "call_tool", short_timeout)
    outcome = await execute("slow", "query_forecast", {**ARGS, "series_id": "slow"})
    assert outcome.is_error and "TimeoutError" in outcome.content
    assert_reaped(server)


@sync
async def test_initialization_timeout_reaps_server(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    pid_file = tmp_path / "init.pid"
    script = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    monkeypatch.setenv("FRIDAY_MCP_COMMAND", json.dumps([sys.executable, "-c", script, str(pid_file)]))
    with pytest.raises(TimeoutError):
        await mcp.call_tool("query_forecast", ARGS, timeout=0.2)
    assert_reaped(pid_file)


@sync
async def test_cancellation_propagates_and_reaps_server(monkeypatch, server):
    from mcp import ClientSession
    original = ClientSession.call_tool
    started = asyncio.Event()

    async def observe(self, *args, **kwargs):
        started.set()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ClientSession, "call_tool", observe)
    task = asyncio.create_task(execute("cancel", "query_forecast", {**ARGS, "series_id": "slow"}))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_reaped(server)
