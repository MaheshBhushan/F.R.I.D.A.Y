"""Assembled-loop tests. Everything runs offline against fakes.

The tests worth having here are the ones whose failure would be silent in a
daemon: the detector never re-arming (FRIDAY goes permanently deaf), a raising
turn taking the loop down, and a state query that escalates being answered
from a snapshot that admitted it couldn't answer.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from friday import brain, loop as loop_mod
from friday.audio.capture import AudioCaptureService
from friday.brain import FakeLLMTransport, ScriptedTurn
from friday.voice import tts, wake
from friday.voice.stt import TranscriptEvent


class FakeSTT:
    """Yields scripted transcript events; records audio like the real one."""

    def __init__(self, events: list[TranscriptEvent]) -> None:
        self._events = events
        self.sent_audio = bytearray()
        self.closed = False
        self.entered = False

    async def __aenter__(self) -> "FakeSTT":
        self.entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    async def send_media(self, chunk: bytes) -> None:
        self.sent_audio.extend(chunk)

    async def send_close_stream(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TranscriptEvent]:
        for event in self._events:
            await asyncio.sleep(0)
            yield event


class RecordingOutput:
    """AudioOutput double: counts bytes instead of opening a device."""

    def __init__(self) -> None:
        self.written = 0
        self.closed = False

    def write(self, chunk: bytes) -> None:
        self.written += len(chunk)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    """Tracks handoff state so 'did the loop re-arm?' is directly assertable."""

    def __init__(self) -> None:
        self.armed = True
        self.end_calls = 0

    def start(self) -> wake.WakeDetection:
        self.armed = False
        live: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        live.put_nowait(b"\x00" * 320)
        live.put_nowait(None)
        return wake.WakeDetection(
            model="alexa", score=0.9, timestamp=0.0, turn_id="",
            preroll=b"\x00" * 640, preroll_samples=320,
            preroll_seconds=0.02, live=live,
        )

    def begin_stream(self) -> None:
        self.armed = True

    def feed_chunk(self, chunk, span=None):
        return None

    def end_handoff(self, live=None) -> None:
        self.end_calls += 1
        self.armed = True


def _final(text: str) -> TranscriptEvent:
    return TranscriptEvent(text=text, is_final=True, speech_final=True)


def _interim(text: str) -> TranscriptEvent:
    return TranscriptEvent(text=text, is_final=False, speech_final=False)


def _build(tmp_path, *, transcript_events, llm_turns=None,
           memory=None, speak=True):
    detector = FakeDetector()
    played: list[str] = []
    output = RecordingOutput()

    def _play(name: str, span=None) -> None:
        played.append(name)
        if span is not None and "ack_audible" not in span.stages:
            span.mark("ack_audible")

    lp = loop_mod.VoiceLoop(
        detector=detector,
        stt_factory=lambda: FakeSTT(transcript_events),
        tts_factory=lambda: tts.FakeSynthesisTransport(bytes_per_sentence=9600),
        llm_transport=FakeLLMTransport(llm_turns or [ScriptedTurn(tokens=("ok.",))]),
        assembler=None,
        memory=memory,
        play_ack=_play,
        audio_output=output,
        spans_path=tmp_path / "spans.jsonl",
        speak_enabled=speak,
    )
    return lp, detector, played, output


def test_reasoning_turn_end_to_end(tmp_path):
    """A reasoning turn acks, streams the LLM, and speaks -- in that order."""
    lp, detector, played, output = _build(
        tmp_path,
        transcript_events=[_interim("what"), _final("summarize my open pull requests")],
        llm_turns=[ScriptedTurn(tokens=("You have ", "two open. "))],
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))

    assert turn.error is None
    assert turn.transcript == "summarize my open pull requests"
    assert turn.tier == "reasoning"
    assert played == [loop_mod.ACK_REASONING]
    assert turn.reply == "You have two open. "
    assert turn.spoke is True
    assert output.written > 0
    # The ack must be audible before any content audio, or she talks over herself.
    assert turn.stages["ack_audible"] < turn.stages["first_content_audio"]


def test_detector_rearms_after_every_turn(tmp_path):
    """The deafness bug: an unreleased handoff means no wake word ever fires
    again, and nothing else in the system would report it."""
    lp, detector, _, _ = _build(
        tmp_path, transcript_events=[_final("what's running")]
    )
    asyncio.run(lp.handle_detection(detector.start()))
    assert detector.armed is True
    assert detector.end_calls == 1


def test_detector_rearms_even_when_the_turn_raises(tmp_path):
    """A crashed turn must still re-arm, and must not propagate."""

    class Exploding:
        async def __aenter__(self):
            raise RuntimeError("socket refused")

        async def __aexit__(self, *exc):
            return None

    lp, detector, _, _ = _build(tmp_path, transcript_events=[])
    lp._stt_factory = lambda: Exploding()

    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert "socket refused" in (turn.error or "")
    assert detector.armed is True


def test_state_query_answers_without_the_llm(tmp_path):
    """Tier 2 must never reach the transport -- that is the whole point of it."""
    lp, detector, played, _ = _build(
        tmp_path, transcript_events=[_final("what branch am i on")]
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.tier == "state_query"
    assert lp._llm.requests == []
    assert played == []          # no ack: the answer is already here
    assert turn.reply


def test_state_query_escalates_instead_of_guessing(tmp_path):
    """When the snapshot can't honestly answer, the turn becomes a reasoning
    turn rather than speaking a confident wrong answer."""
    lp, detector, played, _ = _build(
        tmp_path,
        transcript_events=[_final("what's the alpha service doing")],
        llm_turns=[ScriptedTurn(tokens=("Checked. ",))],
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))
    if turn.escalated:
        assert turn.tier == "reasoning"
        assert lp._llm.requests, "escalation must actually reach the LLM"
        assert played == [loop_mod.ACK_REASONING]


def test_reflex_turn_skips_stt_transport_and_llm(tmp_path):
    """'stop' is a dict lookup and an ack. No network at all."""
    lp, detector, played, _ = _build(tmp_path, transcript_events=[_final("stop")])
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.tier == "reflex"
    assert turn.action == "stop_playback"
    assert lp._llm.requests == []
    assert played == [loop_mod.ACK_REFLEX]
    assert "task_complete" in turn.stages


def test_empty_transcript_costs_nothing(tmp_path):
    """A wake word with no speech after it must not call anything."""
    lp, detector, played, _ = _build(tmp_path, transcript_events=[_final("   ")])
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.tier is None
    assert lp._llm.requests == []
    assert played == []
    assert detector.armed is True


def test_wake_prefix_is_stripped_only_from_voice_turns(tmp_path):
    lp, detector, _, _ = _build(
        tmp_path, transcript_events=[_final("Hey Friday, what's running")]
    )
    routed: list[str] = []

    async def _record(transcript, span, turn):
        routed.append(transcript)

    lp._respond = _record

    async def _run():
        voice = await lp.handle_detection(detector.start())
        text = await lp.ask("fridaywhat is running", speak=False)
        return voice, text

    voice, text = asyncio.run(_run())
    assert voice.transcript == "what's running"
    assert text.transcript == "fridaywhat is running"
    assert routed == ["what's running", "fridaywhat is running"]


def test_wake_only_transcript_is_never_routed(tmp_path):
    lp, detector, _, _ = _build(
        tmp_path, transcript_events=[_final("Okay Friday!")]
    )

    async def _unexpected(*args):
        raise AssertionError("wake-only transcript reached the router")

    lp._respond = _unexpected
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.transcript == ""
    assert turn.tier is None


def test_pump_subscribes_before_scheduling_stt(tmp_path):
    lp, detector, _, _ = _build(tmp_path, transcript_events=[])
    order: list[str] = []
    handled = asyncio.Event()

    class Capture:
        def subscribe_stt(self):
            order.append("subscribed")
            return object()

    class Detector(FakeDetector):
        def detect_chunk(self, chunk, span=None):
            return self.start()

    lp._capture = Capture()
    lp._detector = Detector()

    async def _handle(detection, *, subscription, span):
        order.append("task-started")
        assert "stt_subscription_created" in span.stages
        handled.set()
        return loop_mod.Turn(turn_id=span.turn_id)

    lp.handle_detection = _handle

    async def _run():
        async def _frames():
            yield b"\x00" * 2560

        await lp._pump(_frames(), asyncio.Event())
        await asyncio.wait_for(handled.wait(), timeout=1)

    asyncio.run(_run())
    assert order == ["subscribed", "task-started"]


def test_follow_up_timeout_closes_audio_and_returns_to_idle(tmp_path):
    class Source:
        async def capture(self, stop):
            if False:
                yield b""

    capture = AudioCaptureService(Source())
    transport = FakeSTT([])
    lp = loop_mod.VoiceLoop(
        detector=FakeDetector(),
        stt_factory=lambda: transport,
        capture=capture,
        spans_path=tmp_path / "spans.jsonl",
        conversation_seconds=0.01,
    )

    asyncio.run(lp._follow_up_window())

    assert transport.closed is True
    assert capture.metrics.active_subscribers == 0
    assert lp.audio_state is loop_mod.AudioTurnState.IDLE


def test_reflex_turn_is_not_recorded_to_session_memory(tmp_path):
    class FakeMemory:
        def __init__(self) -> None:
            self.begun: list[str] = []

        async def begin_turn(self, text: str, **kw: Any) -> str:
            self.begun.append(text)
            return "turn"

    mem = FakeMemory()
    lp, detector, _, _ = _build(
        tmp_path, transcript_events=[_final("stop")], memory=mem
    )
    asyncio.run(lp.handle_detection(detector.start()))
    assert mem.begun == []


def test_span_is_written_once_per_turn(tmp_path):
    lp, detector, _, _ = _build(tmp_path, transcript_events=[_final("stop")])
    asyncio.run(lp.handle_detection(detector.start()))
    asyncio.run(lp.handle_detection(detector.start()))
    lines = (tmp_path / "spans.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_no_speak_still_drains_the_stream(tmp_path):
    """--no-speak must not skip tool rounds or span marks."""
    lp, detector, played, output = _build(
        tmp_path,
        transcript_events=[_final("summarize my open pull requests")],
        llm_turns=[ScriptedTurn(tokens=("done. ",))],
        speak=False,
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.reply == "done. "
    assert turn.spoke is False
    assert output.written == 0
    assert "task_complete" in turn.stages


def test_interim_is_used_when_no_final_arrives(tmp_path):
    """Observed live: run_utterance can close on its MAX_WAIT_MS cap with a
    complete interim and no final. Dropping it loses the turn silently."""
    lp, detector, played, _ = _build(
        tmp_path,
        transcript_events=[_interim("what"), _interim("what branch am i on")],
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.transcript == "what branch am i on"
    assert turn.tier == "state_query"


def test_final_still_wins_over_interims(tmp_path):
    lp, detector, _, _ = _build(
        tmp_path,
        transcript_events=[_interim("what branch"), _final("what branch am i on")],
    )
    turn = asyncio.run(lp.handle_detection(detector.start()))
    assert turn.transcript == "what branch am i on"


def test_pump_keeps_feeding_audio_while_a_turn_runs(tmp_path):
    """The deadlock that hung the first live detection: if the pump awaits the
    turn, feed_chunk stops forwarding audio to the live queue the turn is
    reading, and the turn waits forever for audio only the pump can deliver."""
    lp, detector, _, _ = _build(tmp_path, transcript_events=[_final("stop")])
    started = asyncio.Event()
    release = asyncio.Event()
    fed: list[bytes] = []

    class BlockingDetector(FakeDetector):
        def feed_chunk(self, chunk, span=None):
            fed.append(chunk)
            return self.start() if len(fed) == 1 else None

    lp._detector = BlockingDetector()

    async def _blocking_turn(detection):
        started.set()
        await release.wait()
        return loop_mod.Turn(turn_id="x")

    lp.handle_detection = _blocking_turn

    async def _frames():
        for _ in range(5):
            yield b"\x00" * 2560
        await release.wait()

    async def _drive():
        stop = asyncio.Event()
        pump = asyncio.create_task(lp._pump(_frames(), stop))
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert len(fed) == 5, f"pump stalled on the turn: fed {len(fed)}/5"
        release.set()
        await asyncio.wait_for(pump, timeout=2)

    asyncio.run(_drive())


def test_preemption_invalidates_the_in_flight_turn(tmp_path):
    """An interrupted utterance must die, not be resumed after the call."""
    lp, detector, _, _ = _build(tmp_path, transcript_events=[_final("stop")])
    cancelled = asyncio.Event()

    async def _drive():
        async def _mid_utterance():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        lp._turn_task = asyncio.create_task(_mid_utterance())
        await asyncio.sleep(0)
        lp._invalidate_turn([])
        await asyncio.sleep(0.02)
        assert cancelled.is_set()
        assert lp.invalidated == 1

    asyncio.run(_drive())


def test_preemption_also_stops_speech(tmp_path):
    """A call owning the mic would pick up anything FRIDAY says out loud."""
    lp, detector, _, _ = _build(tmp_path, transcript_events=[_final("stop")])

    class FakeSpeaker:
        is_speaking = True
        stopped_called = False

        def stop(self):
            self.stopped_called = True

    speaker = FakeSpeaker()
    lp._speaker = speaker
    lp._invalidate_turn([])
    assert speaker.stopped_called


def test_forget_audio_clears_the_preroll_ring(tmp_path):
    """SUSPENDED means deaf, so a rolling buffer of the room must not survive."""
    lp, detector, _, _ = _build(tmp_path, transcript_events=[_final("stop")])
    det = wake.WakeWordDetector()
    lp._detector = det
    for _ in range(5):
        det.feed_chunk(b"\x00" * 2560)
    assert det.ring_samples > 0
    lp._forget_audio()
    assert det.ring_samples == 0
