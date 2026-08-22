"""Tests for friday.brain: prompt ordering and caching, incremental token
streaming, concurrent tool rounds, and interim-triggered context assembly.

No network: FakeLLMTransport is injected the same way stt.py's fake Transport
is, and state.snapshot/summarize are stubbed so timings are deterministic.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import AsyncIterator, List

import pytest

from friday import brain
from friday.brain import (
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    WORLD_STATE_OPEN,
    AssembledContext,
    ContextAssembler,
    FakeLLMTransport,
    NullMemory,
    ScriptedTurn,
    ToolCall,
    TurnResult,
    build_request,
    complete,
    ordered_blocks,
)
from friday.core.spans import start_turn
from friday.voice.tts import TTSSpeaker


def sync(fn):
    """Run an async test body on a fresh loop -- plain pytest, no plugins."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


FAKE_SNAP = {
    "git": {"name": "friday", "branch": "main", "dirty": True, "root": "/tmp/friday"},
    "tmux_panes": [{"session": "dev", "window": "0:code", "pane": "1", "command": "nvim"}],
    "listening_ports": [{"port": "5173", "process": "node"}],
    "notable_processes": [{"pid": 42, "name": "claude", "cmdline": "claude"}],
    "resources": {"load_avg": [1.0, 0.9, 0.8], "mem_used_gb": 9.1, "mem_total_gb": 31.0,
                  "battery_pct": 88, "ac_online": True, "disk_free_gb": 120.4},
}


def _ctx(world_state: str = "Project: friday @ main (dirty)", memory: str = "") -> AssembledContext:
    return AssembledContext(
        world_state=world_state, memory=memory, assembled_at=time.monotonic(), query="q"
    )


def _span(tmp_path, kind="reasoning"):
    return start_turn(kind, path=tmp_path / "spans.jsonl")


async def _drain(stream: AsyncIterator[str]) -> str:
    return "".join([token async for token in stream])


# --- (1) prompt ordering + caching ------------------------------------------


@sync
async def test_request_is_ordered_static_first_volatile_last():
    transport = FakeLLMTransport([ScriptedTurn(tokens=("ok",))])
    await _drain(complete("what is running", transport, context=_ctx()))

    request = transport.requests[0]
    blocks = ordered_blocks(request)
    print("\nordered request blocks:")
    for i, block in enumerate(blocks):
        print(f"  {i}: {block}")

    tool_idx = [i for i, b in enumerate(blocks) if b.startswith("tool_def:")]
    system_idx = blocks.index("system+cache_control")
    convo_idx = [i for i, b in enumerate(blocks) if b.startswith("user:")]

    # (i) tool defs and system precede all conversation/volatile content
    assert max(tool_idx) < system_idx < min(convo_idx)
    # (ii) the cache breakpoint marks the end of the stable prefix
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["text"] == SYSTEM_PROMPT
    # (iii) the volatile world-state block is the very last block
    assert blocks[-1] == "user:world_state"


@sync
async def test_world_state_uses_summarize_not_a_raw_snapshot_dump():
    assembler = ContextAssembler(
        snapshot_fn=lambda: FAKE_SNAP, summarize_fn=brain.state.summarize
    )
    context = await assembler.get("q")
    request = build_request("what is running", context=context)
    volatile = request["messages"][-1]["content"][-1]["text"]

    assert volatile.startswith(WORLD_STATE_OPEN)
    assert context.world_state in volatile
    # The compact summary, not the JSON blob.
    assert "Project: friday @ main (dirty)" in context.world_state
    assert "{" not in context.world_state and "cmdline" not in context.world_state

    import json
    raw_chars = len(json.dumps(FAKE_SNAP))
    summary_chars = len(context.world_state)
    print(f"\nsummary: {summary_chars} chars (~{summary_chars // 4} tokens) "
          f"vs raw snapshot {raw_chars} chars (~{raw_chars // 4} tokens)")
    assert summary_chars < raw_chars


@sync
async def test_stable_prefix_is_byte_identical_across_turns():
    transport = FakeLLMTransport([ScriptedTurn(tokens=("a",)), ScriptedTurn(tokens=("b",))])
    await _drain(complete("first", transport, context=_ctx("state A")))
    await _drain(complete("second", transport, context=_ctx("state B")))
    first, second = transport.requests
    assert first["tools"] == second["tools"]
    assert first["system"] == second["system"]
    # ...while the volatile tail did change.
    assert first["messages"] != second["messages"]


# --- (2) streaming ----------------------------------------------------------


@sync
async def test_tokens_arrive_before_the_stream_completes(tmp_path):
    span = _span(tmp_path)
    transport = FakeLLMTransport(
        [ScriptedTurn(tokens=tuple(f"tok{i} " for i in range(6)), token_delay=0.02)]
    )
    received: List[tuple[float, str]] = []
    t0 = time.perf_counter()
    stream = complete("talk to me", transport, context=_ctx(), span=span)
    async for token in stream:
        received.append(((time.perf_counter() - t0) * 1000, token))

    assert len(received) == 6
    # The first token landed well before the last -- not one batched dump.
    assert received[0][0] < received[-1][0] / 2
    assert "first_token" in span.stages
    print(f"\nfirst token at {received[0][0]:.1f}ms, last at {received[-1][0]:.1f}ms")
    print("span:", {k: round(v / 1e6, 2) for k, v in span.stages.items()}, "(ms)")


@sync
async def test_token_stream_plugs_into_tts_speak():
    from friday.voice.tts import FakeSynthesisTransport

    class NullOutput:
        def write(self, chunk: bytes) -> None:
            pass

        def close(self) -> None:
            pass

    synth = FakeSynthesisTransport(bytes_per_sentence=320)
    speaker = TTSSpeaker(synth, output=NullOutput())
    transport = FakeLLMTransport([ScriptedTurn(tokens=("The dev ", "server is up. ", "Port 5173."))])
    await speaker.speak(complete("status", transport, context=_ctx()))
    assert synth.synthesized  # speak() consumed complete()'s stream unmodified
    print("\ntts spoke sentences:", synth.synthesized)


# --- (3) parallel tools -----------------------------------------------------


@sync
async def test_three_tool_calls_run_concurrently(tmp_path):
    span = _span(tmp_path)
    calls = tuple(
        ToolCall(id=f"c{i}", name="list_processes", input={"name": f"p{i}"}) for i in range(3)
    )
    starts: List[float] = []

    async def slow_tool(name: str = "") -> str:
        starts.append(time.perf_counter())
        await asyncio.sleep(0.1)
        return f"processes for {name}"

    transport = FakeLLMTransport(
        [ScriptedTurn(tool_calls=calls), ScriptedTurn(tokens=("three ", "answers"))]
    )
    result = TurnResult()
    t0 = time.perf_counter()
    text = await _drain(
        complete(
            "diagnose everything",
            transport,
            context=_ctx(),
            registry={"list_processes": slow_tool},
            span=span,
            result=result,
        )
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert text == "three answers"
    assert len(result.outcomes) == 3
    assert all(not o.is_error for o in result.outcomes)
    # 3 x 100ms tools: concurrent means ~100ms, serial would be ~300ms.
    assert elapsed_ms < 200, elapsed_ms
    assert max(starts) - min(starts) < 0.05
    assert "first_tool_call" in span.stages and "tool_done" in span.stages
    assert span.stages["first_tool_call"] < span.stages["tool_done"]
    print(f"\n3x100ms tools completed in {elapsed_ms:.1f}ms "
          f"(serial would be ~300ms); start spread "
          f"{(max(starts) - min(starts)) * 1000:.2f}ms")
    print("span:", {k: round(v / 1e6, 2) for k, v in span.stages.items()}, "(ms)")


@sync
async def test_tool_results_come_back_in_one_user_message():
    calls = tuple(ToolCall(id=f"c{i}", name="list_processes", input={}) for i in range(3))
    transport = FakeLLMTransport(
        [ScriptedTurn(tool_calls=calls), ScriptedTurn(tokens=("done",))]
    )

    async def tool() -> str:
        return "ok"

    await _drain(
        complete("go", transport, context=_ctx(), registry={"list_processes": tool})
    )
    followup = transport.requests[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-1]["role"] == "user"
    assert len(followup[-1]["content"]) == 3
    assert all(b["type"] == "tool_result" for b in followup[-1]["content"])


@sync
async def test_tool_loop_stops_at_max_rounds():
    calls = (ToolCall(id="c", name="list_processes", input={}),)
    transport = FakeLLMTransport([ScriptedTurn(tool_calls=calls)])

    async def tool() -> str:
        return "ok"

    result = TurnResult()
    await _drain(
        complete(
            "loop forever",
            transport,
            context=_ctx(),
            registry={"list_processes": tool},
            result=result,
        )
    )
    assert result.rounds == MAX_TOOL_ROUNDS


@sync
async def test_denied_tool_is_reported_back_to_the_model():
    calls = (ToolCall(id="c", name="delete_path", input={"path": "/tmp/x"}),)
    transport = FakeLLMTransport(
        [ScriptedTurn(tool_calls=calls), ScriptedTurn(tokens=("I need approval.",))]
    )
    result = TurnResult()
    text = await _drain(
        complete("delete it", transport, context=_ctx(), approve=None, result=result)
    )
    assert result.outcomes[0].is_error
    assert result.outcomes[0].content.startswith("DENIED:")
    assert text == "I need approval."


# --- (4) interim-triggered context assembly ---------------------------------


@sync
async def test_context_assembly_starts_on_interim_and_is_ready_before_the_final(tmp_path):
    span = _span(tmp_path)
    snapshot_calls = []

    def slow_snapshot() -> dict:
        snapshot_calls.append(time.perf_counter())
        time.sleep(0.08)  # stands in for tmux/git/ss subprocess latency
        return FAKE_SNAP

    assembler = ContextAssembler(snapshot_fn=slow_snapshot)

    # interim transcript lands: fire and forget
    assembler.prewarm("what's the dev", span=span)
    # ...more speech arrives while assembly runs in the background
    await asyncio.sleep(0.12)

    # final transcript lands: context must cost ~nothing now
    t0 = time.perf_counter()
    context = await assembler.get("what's the dev server doing", span=span)
    critical_path_ms = (time.perf_counter() - t0) * 1000

    span.mark("stt_final")
    transport = FakeLLMTransport([ScriptedTurn(tokens=("Up on 5173.",))])
    await _drain(complete("what's the dev server doing", transport, context=context, span=span))

    assert context.reused is True
    assert len(snapshot_calls) == 1
    assert critical_path_ms < 5.0, critical_path_ms
    assert span.stages["context_ready"] < span.stages["stt_final"]
    assert span.stages["context_ready"] < span.stages["llm_sent"]
    print(f"\ncontext on the critical path: {critical_path_ms:.3f}ms")
    print("span timeline (ms):",
          {k: round(v / 1e6, 2) for k, v in sorted(span.stages.items(), key=lambda kv: kv[1])})


@sync
async def test_fresh_context_is_reused_under_five_seconds():
    calls = []
    assembler = ContextAssembler(
        snapshot_fn=lambda: (calls.append(1), FAKE_SNAP)[1], stale_ok_seconds=5.0
    )
    first = await assembler.get("q1")
    second = await assembler.get("q2")
    assert len(calls) == 1
    assert second.reused is True
    assert second.world_state == first.world_state
    print(f"\nreused a {(time.monotonic() - first.assembled_at) * 1000:.2f}ms-old context")


@sync
async def test_stale_context_beyond_the_window_is_reassembled():
    calls = []
    assembler = ContextAssembler(
        snapshot_fn=lambda: (calls.append(1), FAKE_SNAP)[1], stale_ok_seconds=0.02
    )
    await assembler.get("q1")
    await asyncio.sleep(0.05)
    again = await assembler.get("q2")
    assert len(calls) == 2
    assert again.reused is False


@sync
async def test_prewarm_is_idempotent_across_repeated_interims():
    calls = []
    assembler = ContextAssembler(snapshot_fn=lambda: (calls.append(1), FAKE_SNAP)[1])
    for interim in ("what", "what's", "what's the", "what's the dev server"):
        assembler.prewarm(interim)
    await assembler.get("what's the dev server doing")
    assert len(calls) == 1


@sync
async def test_null_memory_contributes_nothing_to_the_prompt():
    assembler = ContextAssembler(memory=NullMemory(), snapshot_fn=lambda: FAKE_SNAP)
    context = await assembler.get("q")
    assert context.memory == ""
    request = build_request("q", context=context)
    assert "<recalled>" not in request["messages"][-1]["content"][-1]["text"]


@sync
async def test_injected_memory_lands_after_world_state():
    class StubMemory:
        async def retrieve(self, query: str) -> str:
            return "You asked about the dev server yesterday."

    assembler = ContextAssembler(memory=StubMemory(), snapshot_fn=lambda: FAKE_SNAP)
    context = await assembler.get("q")
    volatile = build_request("q", context=context)["messages"][-1]["content"][-1]["text"]
    assert volatile.index(WORLD_STATE_OPEN) < volatile.index("<recalled>")


@sync
async def test_complete_assembles_context_itself_when_none_is_supplied():
    assembler = ContextAssembler(snapshot_fn=lambda: FAKE_SNAP)
    transport = FakeLLMTransport([ScriptedTurn(tokens=("hi",))])
    await _drain(complete("q", transport, assembler=assembler))
    assert WORLD_STATE_OPEN in transport.requests[0]["messages"][-1]["content"][-1]["text"]


# --- (6) missing key --------------------------------------------------------


def test_require_api_key_exits_naming_the_variable(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        brain._require_api_key()
    assert excinfo.value.code == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_require_api_key_returns_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert brain._require_api_key() == "sk-test"
