"""Hot session-memory persistence, bounds, failure, and prompt integration."""

from __future__ import annotations

import asyncio
import functools
import sqlite3
import time

import pytest

from friday import loop as loop_mod
from friday.brain import FakeLLMTransport, ScriptedTurn, ToolCall
from friday.session_memory import MAX_TURNS, SessionMemory
from friday.tiers.state_query import StateAnswer
from friday.voice import tts


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


async def _complete(mem: SessionMemory, number: int, *, route="reasoning", **kwargs):
    turn_id = await mem.begin_turn(f"user {number}", turn_id=f"t{number}", route_tier=route)
    await mem.complete_turn(turn_id, f"assistant {number}", route_tier=route, **kwargs)


@sync
async def test_empty_memory(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    assert await mem.recent_turns() == []
    assert await mem.context_messages() == []


@sync
async def test_empty_user_text_never_creates_a_pending_turn(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    assert await mem.begin_turn("  \n ") == ""
    assert mem._conn.execute("SELECT count(*) FROM conversation_turns").fetchone()[0] == 0


@sync
async def test_one_completed_turn_is_a_paired_context(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    await _complete(mem, 1)
    assert await mem.context_messages() == [
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
    ]


@sync
async def test_ten_completed_turns_are_retained(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    for i in range(10):
        await _complete(mem, i)
    assert len(await mem.recent_turns()) == 10
    assert len(await mem.context_messages()) == 20


@sync
async def test_eleventh_turn_evicts_only_the_oldest(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    for i in range(11):
        await _complete(mem, i)
    turns = await mem.recent_turns()
    assert [turn.turn_id for turn in turns] == [f"t{i}" for i in range(1, 11)]


@sync
async def test_turns_and_messages_remain_chronological(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    for i in range(3):
        await _complete(mem, i)
    assert [message["content"] for message in await mem.context_messages()] == [
        "user 0", "assistant 0", "user 1", "assistant 1", "user 2", "assistant 2"
    ]


@sync
async def test_pending_and_failed_turns_are_excluded(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    pending = await mem.begin_turn("pending", turn_id="pending")
    failed = await mem.begin_turn("failed", turn_id="failed")
    await mem.fail_turn(failed, "model failed")
    assert pending and await mem.context_messages() == []


@sync
async def test_interrupted_metadata_survives_restart(tmp_path):
    path = tmp_path / "memory.db"
    mem = SessionMemory(path)
    await _complete(mem, 1, interrupted=True)
    await mem.close()
    restored = SessionMemory(path)
    turns = await restored.recent_turns()
    assert turns[0].interrupted is True
    assert "interrupted" in (await restored.context_messages())[1]["content"]


@sync
async def test_restart_restores_active_session_and_recent_turns(tmp_path):
    path = tmp_path / "memory.db"
    first = SessionMemory(path)
    session_id = first.session_id
    for i in range(3):
        await _complete(first, i)
    await first.close()
    second = SessionMemory(path)
    assert second.session_id == session_id
    assert len(await second.recent_turns()) == 3


@sync
async def test_new_session_has_clean_context_but_preserves_old_rows(tmp_path):
    path = tmp_path / "memory.db"
    mem = SessionMemory(path)
    old = mem.session_id
    await _complete(mem, 1)
    new = await mem.new_session()
    assert new != old and await mem.context_messages() == []
    row = mem._conn.execute("SELECT status FROM sessions WHERE id = ?", (old,)).fetchone()
    assert row["status"] == "closed"
    assert mem._conn.execute(
        "SELECT count(*) FROM conversation_turns WHERE session_id = ?", (old,)
    ).fetchone()[0] == 1


@sync
async def test_limit_zero_disables_injection_but_persists(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db", turn_limit=0)
    await _complete(mem, 1)
    assert await mem.context_messages() == []
    assert mem._conn.execute("SELECT count(*) FROM conversation_turns").fetchone()[0] == 1


@sync
async def test_explicit_limit_is_applied(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db", turn_limit=10)
    for i in range(12):
        await _complete(mem, i)
    assert len(await mem.recent_turns()) == 10


@sync
async def test_recent_turns_call_can_request_a_smaller_window(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    for i in range(5):
        await _complete(mem, i)
    assert [turn.turn_id for turn in await mem.recent_turns(2)] == ["t3", "t4"]


@pytest.mark.parametrize("value", [-1, MAX_TURNS + 1])
def test_invalid_turn_limit_is_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        SessionMemory(tmp_path / "memory.db", turn_limit=value)


def test_invalid_environment_limit_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIDAY_SESSION_MEMORY_TURNS", "unbounded")
    with pytest.raises(ValueError):
        SessionMemory(tmp_path / "memory.db")


@sync
async def test_context_budget_drops_oldest_complete_turns(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db", context_chars=24)
    for i in range(3):
        await _complete(mem, i)
    messages = await mem.context_messages()
    assert [item["content"] for item in messages] == ["user 2", "assistant 2"]


@sync
async def test_oversized_newest_turn_produces_no_partial_messages(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db", context_chars=5)
    await _complete(mem, 1)
    assert await mem.context_messages() == []


@sync
async def test_unicode_and_multiline_round_trip(tmp_path):
    path = tmp_path / "memory.db"
    mem = SessionMemory(path)
    turn = await mem.begin_turn("Grüße 👋", turn_id="unicode")
    await mem.complete_turn(turn, "line one\nline two — ✓")
    await mem.close()
    restored = SessionMemory(path)
    messages = await restored.context_messages()
    assert messages[0]["content"] == "Grüße 👋"
    assert messages[1]["content"] == "line one\nline two — ✓"


@sync
async def test_route_and_optional_metadata_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    mem = SessionMemory(path)
    turn = await mem.begin_turn("inspect", turn_id="meta", route_tier="reasoning")
    await mem.complete_turn(
        turn, "done", route_tier="reasoning", metadata={"tool_name": "list_processes"}
    )
    await mem.close()
    restored = SessionMemory(path)
    item = (await restored.recent_turns())[0]
    assert item.route_tier == "reasoning"
    assert item.metadata == {"tool_name": "list_processes"}


@sync
async def test_sqlite_write_failure_keeps_hot_memory_alive(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    mem._conn.close()
    turn = await mem.begin_turn("still work", turn_id="degraded")
    assert await mem.complete_turn(turn, "without persistence") is True
    assert (await mem.context_messages())[1]["content"] == "without persistence"


@sync
async def test_corrupt_database_degrades_without_constructor_failure(tmp_path):
    path = tmp_path / "memory.db"
    path.write_bytes(b"not sqlite")
    mem = SessionMemory(path)
    assert mem._conn is None
    turn = await mem.begin_turn("hello")
    assert await mem.complete_turn(turn, "hi") is True


@sync
async def test_concurrent_completions_are_atomic_and_paired(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db", turn_limit=20)

    async def write(i):
        turn = await mem.begin_turn(f"u{i}", turn_id=f"c{i}")
        await asyncio.sleep(0)
        await mem.complete_turn(turn, f"a{i}")

    await asyncio.gather(*(write(i) for i in range(20)))
    turns = await mem.recent_turns()
    assert len(turns) == 20
    assert all(turn.user.content[1:] == turn.assistant.content[1:] for turn in turns)


def test_schema_migration_is_recorded(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    versions = mem._conn.execute(
        "SELECT version FROM session_schema_migrations ORDER BY version"
    ).fetchall()
    assert [row[0] for row in versions] == [1]


@sync
async def test_restart_after_new_session_restores_only_new_active_context(tmp_path):
    path = tmp_path / "memory.db"
    mem = SessionMemory(path)
    await _complete(mem, 1)
    active = await mem.new_session()
    await _complete(mem, 2)
    await mem.close()
    restored = SessionMemory(path)
    assert restored.session_id == active
    assert [turn.turn_id for turn in await restored.recent_turns()] == ["t2"]


@sync
async def test_context_hot_path_latency_is_submillisecond_typically(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    for i in range(10):
        await _complete(mem, i)
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        await mem.context_messages()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    assert samples[94] < 5.0


class _RecordingOutput:
    def write(self, data):
        pass

    def close(self):
        pass


def _voice_loop(tmp_path, mem, transport):
    return loop_mod.VoiceLoop(
        tts_factory=lambda: tts.FakeSynthesisTransport(),
        llm_transport=transport,
        session_memory=mem,
        play_ack=lambda *args: None,
        audio_output=_RecordingOutput(),
        spans_path=tmp_path / "spans.jsonl",
        speak_enabled=False,
        follow_up=False,
    )


@sync
async def test_three_turn_history_is_injected_without_current_request_duplication(
    tmp_path, monkeypatch
):
    mem = SessionMemory(tmp_path / "memory.db")
    monkeypatch.setattr(
        loop_mod.state_query,
        "answer",
        lambda *args, **kwargs: StateAnswer(
            "Codex is running the authentication tests.", True, False, "agent_doing"
        ),
    )
    transport = FakeLLMTransport([
        ScriptedTurn(tokens=("Two tests failed.",)),
        ScriptedTurn(tokens=("I'll tell Codex to fix those tests.",)),
    ])
    voice = _voice_loop(tmp_path, mem, transport)
    await voice.ask("What's Codex doing?", speak=False)
    await voice.ask("Did anything fail?", speak=False)
    await voice.ask("Tell it to fix them.", speak=False)
    messages = transport.requests[1]["messages"]
    assert [item["content"] for item in messages[:-1]] == [
        "What's Codex doing?", "Codex is running the authentication tests.",
        "Did anything fail?", "Two tests failed.",
    ]
    current_blocks = messages[-1]["content"]
    assert sum(block.get("text") == "Tell it to fix them." for block in current_blocks) == 1


@sync
async def test_tool_backed_semantic_response_is_stored(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    transport = FakeLLMTransport([
        ScriptedTurn(tool_calls=(ToolCall("call", "list_processes", {}),)),
        ScriptedTurn(tokens=("Codex is running tests.",)),
    ])
    voice = _voice_loop(tmp_path, mem, transport)
    await voice.ask("inspect Codex deeply", speak=False)
    assert (await mem.context_messages())[-1]["content"] == "Codex is running tests."


@sync
async def test_reflex_and_wake_only_do_not_pollute_loop_memory(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    transport = FakeLLMTransport([ScriptedTurn(tokens=("unused",))])
    voice = _voice_loop(tmp_path, mem, transport)
    await voice.ask("stop", speak=False)
    assert await mem.context_messages() == []


@sync
async def test_reasoning_failure_remains_pending_or_failed_not_completed(tmp_path):
    class BrokenTransport:
        async def stream(self, request):
            raise RuntimeError("provider down")
            yield

    mem = SessionMemory(tmp_path / "memory.db")
    voice = _voice_loop(tmp_path, mem, BrokenTransport())
    turn = await voice.ask("explain this failure", speak=False)
    assert "provider down" in (turn.error or "")
    assert await mem.context_messages() == []
    status = mem._conn.execute("SELECT status FROM conversation_turns").fetchone()[0]
    assert status == "failed"


@sync
async def test_restart_history_reaches_next_provider_request(tmp_path):
    path = tmp_path / "memory.db"
    first = SessionMemory(path)
    for i in range(3):
        await _complete(first, i)
    await first.close()
    restored = SessionMemory(path)
    transport = FakeLLMTransport([ScriptedTurn(tokens=("follow-up",))])
    voice = _voice_loop(tmp_path, restored, transport)
    await voice.ask("what about those?", speak=False)
    assert len(transport.requests[0]["messages"]) == 7


@sync
async def test_memory_failure_does_not_crash_voice_loop(tmp_path):
    class BrokenMemory:
        async def begin_turn(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk unavailable")

        async def context_messages(self):
            raise sqlite3.OperationalError("disk unavailable")

    transport = FakeLLMTransport([ScriptedTurn(tokens=("still answered",))])
    voice = _voice_loop(tmp_path, BrokenMemory(), transport)
    turn = await voice.ask("answer anyway", speak=False)
    assert turn.reply == "still answered"


@sync
async def test_acknowledgement_is_never_an_assistant_message(tmp_path):
    mem = SessionMemory(tmp_path / "memory.db")
    transport = FakeLLMTransport([ScriptedTurn(tokens=("semantic answer",))])
    voice = _voice_loop(tmp_path, mem, transport)
    await voice.ask("do some reasoning", speak=False)
    contents = [item["content"] for item in await mem.context_messages()]
    assert contents == ["do some reasoning", "semantic answer"]
    assert "Checking." not in contents
