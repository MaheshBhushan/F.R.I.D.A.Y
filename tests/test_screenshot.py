"""screenshot tool: image block shape, backend fallbacks, and the kill switch."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from friday import permissions
from friday.tools import TOOL_SPECS, TOOLS, ToolOutcome, execute
from friday.tools import screenshot as shot

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []

    async def capture(target, out: Path):
        calls.append(target)
        out.write_bytes(TINY_PNG)
        return "fake"

    async def no_shrink(png, jpg):
        return False

    monkeypatch.setattr(shot, "_capture_png", capture)
    monkeypatch.setattr(shot, "_shrink", no_shrink)
    monkeypatch.delenv("FRIDAY_SCREENSHOT", raising=False)
    return calls


def test_registered_as_read_only_tool():
    assert TOOLS["screenshot"] is shot.screenshot
    assert permissions.risk_of("screenshot") is permissions.Risk.READ_ONLY
    spec = next(s for s in TOOL_SPECS if s["name"] == "screenshot")
    assert spec["input_schema"]["properties"]["target"]["enum"] == list(shot.TARGETS)


def test_returns_image_then_text_block(fake_backend):
    blocks = asyncio.run(shot.screenshot("window"))
    assert fake_backend == ["window"]
    assert blocks[0]["type"] == "image"
    src = blocks[0]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/png"
    assert base64.b64decode(src["data"]) == TINY_PNG
    assert blocks[1]["type"] == "text" and "fake" in blocks[1]["text"]


def test_jpeg_when_a_shrinker_is_available(fake_backend, monkeypatch):
    async def shrink(png, jpg):
        jpg.write_bytes(b"\xff\xd8jpeg")
        return True

    monkeypatch.setattr(shot, "_shrink", shrink)
    blocks = asyncio.run(shot.screenshot())
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(blocks[0]["source"]["data"]) == b"\xff\xd8jpeg"


def test_execute_passes_block_list_through_as_tool_result(fake_backend):
    outcome = asyncio.run(execute("tu_1", "screenshot", {}))
    assert not outcome.is_error
    block = outcome.to_result_block()
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu_1"
    assert isinstance(block["content"], list)
    assert block["content"][0]["type"] == "image"


def test_kill_switch_refuses_before_capturing(fake_backend, monkeypatch):
    monkeypatch.setenv("FRIDAY_SCREENSHOT", "0")
    outcome = asyncio.run(execute("tu_2", "screenshot", {}))
    assert outcome.is_error and outcome.content.startswith("DENIED")
    assert fake_backend == []


def test_bad_target_is_an_argument_error(fake_backend):
    outcome = asyncio.run(execute("tu_3", "screenshot", {"target": "keyboard"}))
    assert outcome.is_error and "target must be one of" in outcome.content


def test_missing_backend_is_a_tool_error(monkeypatch):
    monkeypatch.delenv("FRIDAY_SCREENSHOT", raising=False)
    monkeypatch.setattr(shot.shutil, "which", lambda name: None)
    outcome = asyncio.run(execute("tu_4", "screenshot", {}))
    assert outcome.is_error and "no working screenshot backend" in outcome.content


def test_string_outcomes_still_work():
    assert ToolOutcome("x", "read_file", "hello").to_result_block()["content"] == "hello"
