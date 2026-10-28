"""Deterministic multi-round timing, independent of wall-clock performance."""

import asyncio
import json

from friday import brain
from friday.core.spans import TurnSpan
from friday.scripts import percentiles


def test_model_wait_excludes_tools_and_token_consumer_time(monkeypatch, tmp_path):
    clock = [0]
    monkeypatch.setattr(brain.time, "perf_counter_ns", lambda: clock[0])
    span = TurnSpan("reasoning", path=tmp_path / "spans.jsonl")

    class Transport:
        def __init__(self):
            self.round = 0

        async def stream(self, request):
            self.round += 1
            clock[0] += 10_000_000
            yield brain.LLMEvent(kind="text", text="token")
            clock[0] += 20_000_000
            calls = (brain.ToolCall("f", "query_forecast", {}),) if self.round < 3 else ()
            yield brain.LLMEvent(kind="turn_end", tool_calls=calls, assistant_content=[])

    async def tool():
        clock[0] += 100_000_000
        return "forecast"

    async def run():
        async for _token in brain.complete("forecast", Transport(), span=span,
                                          registry={"query_forecast": tool}):
            clock[0] += 1_000_000_000  # Time spent speaking/consuming each token.
    asyncio.run(run())
    assert span.durations_ns == {"llm_wait": 90_000_000, "tools": 200_000_000}
    span.write()
    record = json.loads(span.path.read_text())
    assert record["durations_ns"] == span.durations_ns
    summary = percentiles.summarize([record, {"turn_kind": "reasoning", "stages": {}}])
    assert [(r["name"], r["p50_ms"], r["count"]) for r in summary["durations"]] == [
        ("llm_wait", 90.0, 1), ("tools", 200.0, 1),
    ]
    assert "accumulated durations" in percentiles.format_table(summary)


def test_cancelled_tool_round_still_records_elapsed_time(monkeypatch, tmp_path):
    clock = [0]
    monkeypatch.setattr(brain.time, "perf_counter_ns", lambda: clock[0])
    span = TurnSpan("reasoning", path=tmp_path / "spans.jsonl")

    async def run():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def tool():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(brain._run_tool_round(
            (brain.ToolCall("f", "query_forecast", {}),), None, {"query_forecast": tool}, span
        ))
        await started.wait()
        clock[0] += 42
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert cleaned.is_set()

    asyncio.run(run())
    assert span.durations_ns == {"tools": 42}
    assert span.stages["tool_done"] == 42
