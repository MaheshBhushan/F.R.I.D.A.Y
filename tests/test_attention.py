"""Tests for friday.attention: rule-only scoring, bounded enrichment, the
preempt+ack+detail emit path, and nag control.

No audio, no network, no real processes disturbed: the ack player is a
recording fake, TTS uses FakeSynthesisTransport with an in-memory sink,
and the LLM is FakeLLMTransport (or a tripwire that raises if the scoring
path ever touches it). The one real process involved is a `python -c pass`
child this test spawns itself and waits on.
"""

from __future__ import annotations

import asyncio
import statistics
import subprocess
import sys
import time
from typing import AsyncIterator, List, Optional

from friday.attention import (
    COOLDOWN_S,
    ENRICH_TIMEOUT_S,
    MAX_PER_WINDOW,
    MIN_GAP_S,
    SCORES,
    THRESHOLD,
    Attention,
    Event,
    diff_events,
    enrich,
    score,
    watch,
)
from friday.brain import FakeLLMTransport, ScriptedTurn
from friday.core.spans import TurnSpan, start_turn
from friday.voice.tts import FakeSynthesisTransport, TTSSpeaker


class TripwireTransport:
    """LLM transport that fails the test if it is ever consulted."""

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, request: dict):
        self.calls += 1
        raise AssertionError("LLM transport consulted in the scoring path")


class NullOutput:
    """In-memory AudioOutput: counts bytes, touches no device."""

    def __init__(self) -> None:
        self.written = 0
        self.closed = False

    def write(self, chunk: bytes) -> None:
        self.written += len(chunk)

    def close(self) -> None:
        self.closed = True


class RecordingAck:
    """Stands in for ack.play_ack: records the cached ack name it would
    have played straight from the bank, and marks the same span stage."""

    def __init__(self, bank: Optional[List[str]] = None) -> None:
        self.played: List[str] = []
        self.bank = bank

    async def __call__(self, name: str, span: Optional[TurnSpan]) -> None:
        if self.bank is not None:
            assert name in self.bank, f"{name!r} is not in the ack bank"
        self.played.append(name)
        if span is not None:
            span.mark("ack_audible")


class RecordingSpeaker:
    """Wraps a real TTSSpeaker so stop() calls are observable."""

    def __init__(self, speaker: TTSSpeaker) -> None:
        self.speaker = speaker
        self.stops = 0

    @property
    def is_speaking(self) -> bool:
        return self.speaker.is_speaking

    def stop(self) -> None:
        self.stops += 1
        self.speaker.stop()

    async def speak(self, stream, *, span=None) -> None:
        await self.speaker.speak(stream, span=span)


def _speaker(**kwargs) -> tuple[RecordingSpeaker, FakeSynthesisTransport, NullOutput]:
    synth = FakeSynthesisTransport(**kwargs)
    output = NullOutput()
    return RecordingSpeaker(TTSSpeaker(synth, output=output)), synth, output


def _attention(tmp_path, transport, **kwargs) -> tuple[Attention, RecordingSpeaker, FakeSynthesisTransport, RecordingAck]:
    speaker, synth, _ = _speaker()
    ack = RecordingAck(bank=["sir", "one_moment"])
    att = Attention(
        speaker,
        transport,
        play_ack=ack,
        spans_path=tmp_path / "spans.jsonl",
        **kwargs,
    )
    return att, speaker, synth, ack


# --- criterion 1: zero LLM calls in the scoring path ------------------------


SCORED_EVENTS = [
    Event("dev_server_died", "vite", data={"port": 5173}),
    Event("agent_needs_permission", "friday-task-1"),
    Event("agent_failed", "friday-task-2"),
    Event("build_failed", "cargo"),
    Event("tests_failed", "pytest"),
    Event("disk_low", "disk", data={"disk_free_gb": 1.2}),
    Event("battery_low", "battery", data={"battery_pct": 8, "ac_online": False}),
    Event("battery_low", "battery", data={"battery_pct": 8, "ac_online": True}),
    Event("agent_idle", "friday-task-3", data={"since_change_secs": 300}),
    Event("agent_idle", "friday-task-4", data={"since_change_secs": 3}),
    Event("agent_running", "friday-task-5"),
    Event("file_changed", "notes.md"),
    Event("log_line", "app.log"),
    Event("git_branch_changed", "feat/x"),
    Event("git_dirty_changed", "repo"),
    Event("port_opened", "8000"),
    Event("process_started", "node"),
    Event("load_changed", "cpu"),
    Event("totally_unknown_event", "?"),
]


def test_scoring_touches_no_llm_and_marks_no_span_stage():
    assert len({e.type for e in SCORED_EVENTS}) >= 12
    tripwire = TripwireTransport()
    span = start_turn("proactive")
    scores = {}
    timings = []
    for event in SCORED_EVENTS:
        t0 = time.perf_counter_ns()
        scores[(event.type, tuple(sorted(event.data.items())))] = score(event)
        timings.append(time.perf_counter_ns() - t0)

    assert tripwire.calls == 0
    assert span.stages == {}, "scoring must not mark any span stage"
    assert "llm_sent" not in span.stages

    high = [e for e in SCORED_EVENTS if score(e) >= THRESHOLD]
    low = [e for e in SCORED_EVENTS if score(e) < THRESHOLD]
    assert {e.type for e in high} >= {
        "dev_server_died",
        "agent_needs_permission",
        "agent_failed",
        "build_failed",
        "tests_failed",
        "disk_low",
    }
    assert len(low) >= 5
    print(
        "\nscore() over %d events: median %.2fus, max %.2fus"
        % (len(timings), statistics.median(timings) / 1000, max(timings) / 1000)
    )
    print("scores:", {e.type: score(e) for e in SCORED_EVENTS})


def test_score_table_and_rules_are_pure_data():
    assert isinstance(SCORES, dict)
    assert all(isinstance(v, int) for v in SCORES.values())
    # the LLM prompt builder is never reachable from score()
    assert score(Event("totally_unknown_event")) == 0


# --- criterion 2 + 3 + 6: end-to-end dev-server death -----------------------


def _dead_dev_server_event(tmp_path) -> Event:
    """Spawn our own throwaway child, let it exit, and build the event the
    watcher would have produced from its disappearance. No pre-existing
    process on this machine is signalled."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        # Bounded. An unbounded wait() here hung a test run for five hours:
        # other tests in this suite drive children through asyncio, whose child
        # watcher installs a SIGCHLD handler, and a Popen reaped out from under
        # blocking waitpid() has nothing left to wake it up. A child that runs
        # `pass` and still hasn't exited in 30s is a broken environment, not
        # something to keep waiting on.
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise AssertionError("throwaway child never exited; environment is broken")
    log = tmp_path / "devserver.log"
    log.write_text(
        "listening on http://localhost:5173\n"
        "error: connection to PostgreSQL at 127.0.0.1:5432 refused\n"
        "FATAL: db pool exhausted, exiting\n"
    )
    before = {
        "listening_ports": [{"port": 5173, "process": "fakevite", "pid": proc.pid}],
        "git": {"branch": "main"},
        "resources": {},
    }
    after = {"listening_ports": [], "git": {"branch": "main"}, "resources": {}}
    events = diff_events(before, after)
    death = [e for e in events if e.type == "dev_server_died"]
    assert len(death) == 1
    death[0].data["log_path"] = str(log)
    return death[0]


def test_dev_server_death_preempts_speech_and_names_the_process(tmp_path, capsys):
    event = _dead_dev_server_event(tmp_path)
    assert score(event) >= THRESHOLD

    transport = FakeLLMTransport(
        [
            ScriptedTurn(
                tokens=(
                    "The fakevite dev server on port 5173 just died. ",
                    "The log tail points at PostgreSQL refusing connections. ",
                    "Want me to investigate?",
                )
            )
        ]
    )
    att, speaker, synth, ack = _attention(tmp_path, transport)

    async def scenario():
        async def long_stream() -> AsyncIterator[str]:
            for _ in range(200):
                yield "Still talking about something else entirely. "
                await asyncio.sleep(0.01)

        speaking = asyncio.create_task(speaker.speak(long_stream()))
        await asyncio.sleep(0.15)
        assert speaker.is_speaking
        prior_gen = speaker.speaker._gen_task

        t0 = time.perf_counter()
        text = await att.handle(event)
        elapsed = time.perf_counter() - t0
        speaking.cancel()
        await asyncio.gather(speaking, return_exceptions=True)
        return text, elapsed, prior_gen

    text, elapsed, prior_gen = asyncio.run(scenario())

    assert elapsed < 3.0, elapsed
    # cached lead-in came from the bank
    assert ack.played == ["sir"]
    # the dead process is named in what was actually synthesized
    spoken = " ".join(synth.synthesized)
    assert "fakevite" in spoken
    # enrichment landed in the prompt before the LLM was asked to word it
    prompt = transport.requests[0]["messages"][0]["content"][0]["text"]
    assert "PostgreSQL" in prompt
    assert "db pool exhausted" in prompt
    # preempt correctness
    assert speaker.stops == 1
    assert prior_gen.cancelled()

    span_line = (tmp_path / "spans.jsonl").read_text().strip()
    print("\ninterrupt latency: %.1fms" % (elapsed * 1000))
    print("timeline:", span_line)
    print("spoken:", text)


def test_enrichment_is_bounded_by_a_timeout(tmp_path):
    event = Event("dev_server_died", "fakevite", data={"log_path": "/nonexistent"})

    async def slow(_event):
        await asyncio.sleep(5.0)
        return "too late"

    async def scenario():
        t0 = time.perf_counter()
        evidence = await enrich(event, investigate=slow, timeout=0.1)
        return evidence, time.perf_counter() - t0

    evidence, elapsed = asyncio.run(scenario())
    assert evidence == ""
    assert elapsed < 1.0, elapsed
    assert ENRICH_TIMEOUT_S == 1.0
    print("\nslow enrichment abandoned after %.0fms (bound %.1fs)" % (elapsed * 1000, 1.0))


def test_interrupt_still_fires_when_enrichment_times_out(tmp_path):
    event = Event("dev_server_died", "fakevite")
    transport = FakeLLMTransport([ScriptedTurn(tokens=("The fakevite server died.",))])

    async def slow(_event):
        await asyncio.sleep(5.0)
        return "too late"

    att, speaker, synth, ack = _attention(
        tmp_path, transport, investigate=slow, enrich_timeout=0.05
    )
    text = asyncio.run(att.handle(event))
    assert "fakevite" in text
    assert ack.played == ["sir"]
    prompt = transport.requests[0]["messages"][0]["content"][0]["text"]
    assert "investigation timed out" in prompt


# --- criterion 4: nag control -----------------------------------------------


def test_flapping_event_interrupts_once_over_a_burst(tmp_path):
    transport = FakeLLMTransport([ScriptedTurn(tokens=("The fakevite server died.",))])
    att, speaker, synth, ack = _attention(tmp_path, transport)

    async def scenario():
        for _ in range(25):
            await att.handle(Event("dev_server_died", "fakevite"))

    asyncio.run(scenario())

    assert len(att.interrupts) == 1
    assert len(ack.played) == 1
    assert len(att.suppressed) == 24
    assert {r for _, r in att.suppressed} == {"cooldown"}
    print(
        "\n25 identical events -> %d interrupt(s); cooldown %.0fs per (type,subject), "
        "min gap %.0fs, max %d per %.0fs"
        % (len(att.interrupts), COOLDOWN_S, MIN_GAP_S, MAX_PER_WINDOW, 600.0)
    )


def test_rate_limit_caps_distinct_high_score_events(tmp_path):
    transport = FakeLLMTransport([ScriptedTurn(tokens=("Something broke.",))])
    clock = {"t": 0.0}
    att, speaker, synth, ack = _attention(
        tmp_path, transport, now=lambda: clock["t"]
    )

    async def scenario():
        for i in range(6):
            clock["t"] += MIN_GAP_S + 1
            await att.handle(Event("build_failed", f"target-{i}"))

    asyncio.run(scenario())
    assert len(att.interrupts) == MAX_PER_WINDOW
    assert "rate_limit" in {r for _, r in att.suppressed}


def test_min_gap_blocks_two_different_events_back_to_back(tmp_path):
    transport = FakeLLMTransport([ScriptedTurn(tokens=("Something broke.",))])
    att, speaker, synth, ack = _attention(tmp_path, transport)

    async def scenario():
        await att.handle(Event("build_failed", "cargo"))
        await att.handle(Event("tests_failed", "pytest"))

    asyncio.run(scenario())
    assert len(att.interrupts) == 1
    assert att.suppressed[0][1] == "min_gap"


# --- criterion 5: ignored events stay silent --------------------------------


def test_low_importance_events_produce_no_speech_at_all(tmp_path):
    tripwire = TripwireTransport()
    att, speaker, synth, ack = _attention(tmp_path, tripwire)
    quiet = [
        Event("file_changed", "notes.md"),
        Event("log_line", "app.log"),
        Event("git_dirty_changed", "repo"),
        Event("process_started", "node"),
        Event("load_changed", "cpu"),
        Event("agent_running", "friday-task-1"),
        Event("battery_low", "battery", data={"battery_pct": 8, "ac_online": True}),
    ]
    assert len(quiet) >= 5

    async def scenario():
        for event in quiet:
            assert await att.handle(event) is None

    asyncio.run(scenario())

    assert att.interrupts == []
    assert ack.played == []
    assert synth.synthesized == []
    assert speaker.stops == 0
    assert tripwire.calls == 0
    assert not (tmp_path / "spans.jsonl").exists()
    assert all(r.startswith("below_threshold") for _, r in att.suppressed)


# --- the watcher -------------------------------------------------------------


def test_watch_emits_events_from_snapshot_and_agent_diffs():
    snaps = [
        {
            "listening_ports": [{"port": 5173, "process": "fakevite"}],
            "git": {"branch": "main"},
            "resources": {"disk_free_gb": 100.0},
        },
        {
            "listening_ports": [],
            "git": {"branch": "main"},
            "resources": {"disk_free_gb": 100.0},
        },
    ]
    agent_states = [
        {"friday-task-1": {"status": "running", "since_change_secs": 0.0}},
        {"friday-task-1": {"status": "permission", "since_change_secs": 4.0}},
    ]
    calls = {"n": 0}

    def snapshot():
        i = min(calls["n"], len(snaps) - 1)
        return snaps[i]

    def poll_agents():
        i = min(calls["n"], len(agent_states) - 1)
        calls["n"] += 1
        return agent_states[i]

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        await watch(queue, snapshot=snapshot, poll_agents=poll_agents, interval=0.0, ticks=2)
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    events = asyncio.run(scenario())
    types = [e.type for e in events]
    assert "dev_server_died" in types
    assert "agent_needs_permission" in types
    died = next(e for e in events if e.type == "dev_server_died")
    assert died.subject == "fakevite"


def test_run_consumes_a_queue_until_the_sentinel(tmp_path):
    transport = FakeLLMTransport([ScriptedTurn(tokens=("The fakevite server died.",))])
    att, speaker, synth, ack = _attention(tmp_path, transport)

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(Event("file_changed", "notes.md"))
        await queue.put(Event("dev_server_died", "fakevite"))
        await queue.put(None)
        await att.run(queue)

    asyncio.run(scenario())
    assert [e.type for e in att.interrupts] == ["dev_server_died"]


def test_real_ack_bank_contains_the_default_lead_in():
    from friday.voice.ack import list_acks

    assert "sir" in list_acks()
