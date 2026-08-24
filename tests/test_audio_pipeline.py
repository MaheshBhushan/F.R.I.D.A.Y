"""Deterministic races across capture, wake detection, STT, and routing."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any, AsyncIterator, Optional

import pytest

from friday import brain, loop as loop_mod
from friday.audio.capture import FRAME_BYTES, AudioCaptureService, AudioMetrics
from friday.audio.manager import AudioResourceManager
from friday.core.spans import TurnSpan
from friday.voice import stt, tts, wake
from friday.voice.stt import TranscriptEvent


class Source:
    async def capture(self, stop):
        if False:
            yield b""


class Detector:
    def __init__(self, wake_on: Optional[int] = None) -> None:
        self.wake_on = wake_on
        self.calls = 0
        self.armed = 0

    def detect_chunk(self, chunk: bytes, span=None):
        self.calls += 1
        if self.calls != self.wake_on:
            return None
        return _detection()

    def feed_chunk(self, chunk: bytes, span=None):
        return self.detect_chunk(chunk, span)

    def begin_stream(self) -> None:
        self.armed += 1

    def end_handoff(self, live=None) -> None:
        self.armed += 1


class CountVAD:
    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.seen = 0

    def feed(self, chunk: bytes) -> bool:
        self.seen += 1
        return self.seen >= self.frames


class ScriptedSTT:
    def __init__(
        self,
        events: tuple[TranscriptEvent, ...] = (),
        *,
        expected_audio: int = 0,
        enter_delay: float = 0,
        block_enter: bool = False,
        fail_enter: bool = False,
    ) -> None:
        self.events = events
        self.expected_audio = expected_audio
        self.enter_delay = enter_delay
        self.block_enter = block_enter
        self.fail_enter = fail_enter
        self.sent: list[bytes] = []
        self.enter_started = asyncio.Event()
        self.entered = asyncio.Event()
        self.audio_ready = asyncio.Event()
        self.closed = False
        self.enter_started_at = 0.0
        self.entered_at = 0.0

    async def __aenter__(self):
        self.enter_started_at = time.monotonic()
        self.enter_started.set()
        if self.fail_enter:
            raise ConnectionError("scripted connect failure")
        if self.block_enter:
            await asyncio.Event().wait()
        if self.enter_delay:
            await asyncio.sleep(self.enter_delay)
        self.entered_at = time.monotonic()
        self.entered.set()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.closed = True

    async def send_media(self, chunk: bytes) -> None:
        self.sent.append(chunk)
        if len(self.sent) >= self.expected_audio:
            self.audio_ready.set()

    async def send_close_stream(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TranscriptEvent]:
        if self.expected_audio:
            await self.audio_ready.wait()
        if not self.events:
            await asyncio.Event().wait()
        for event in self.events:
            await asyncio.sleep(0)
            yield event


class Speaker:
    def __init__(self) -> None:
        self.is_speaking = True
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        self.is_speaking = False


class Sink:
    def write(self, chunk: bytes) -> None:
        pass

    def close(self) -> None:
        pass


def _detection() -> wake.WakeDetection:
    live: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    live.put_nowait(None)
    return wake.WakeDetection(
        model="friday",
        score=0.99,
        timestamp=0.0,
        turn_id="",
        preroll=b"",
        preroll_samples=0,
        preroll_seconds=0,
        live=live,
    )


def _frame(number: int) -> bytes:
    return number.to_bytes(2, "little") * (FRAME_BYTES // 2)


def _partial(text: str) -> TranscriptEvent:
    return TranscriptEvent(text, is_final=False)


def _final(text: str) -> TranscriptEvent:
    return TranscriptEvent(text, is_final=True, speech_final=True)


def _loop(tmp_path, detector, capture, factory, *, conversation_seconds=0):
    played: list[str] = []

    def play_ack(name: str, span: TurnSpan) -> None:
        played.append(name)
        span.mark("ack_audible")

    lp = loop_mod.VoiceLoop(
        detector=detector,
        stt_factory=factory,
        llm_transport=brain.FakeLLMTransport(
            [brain.ScriptedTurn(tokens=("Working.",))]
        ),
        capture=capture,
        play_ack=play_ack,
        spans_path=tmp_path / "spans.jsonl",
        speak_enabled=False,
        conversation_seconds=conversation_seconds,
    )
    return lp, played


def test_delayed_connect_preserves_exact_audio_and_routes_final(tmp_path, monkeypatch):
    async def run():
        capture = AudioCaptureService(Source())
        detector = Detector(wake_on=2)
        transport = ScriptedSTT(
            (_partial("Friday check"), _final("Friday check what Codex is doing")),
            expected_audio=16,
            enter_delay=0.5,
        )
        monkeypatch.setattr(stt, "LocalVAD", lambda: CountVAD(16))
        lp, played = _loop(tmp_path, detector, capture, lambda: transport)
        pump = asyncio.create_task(lp._pump(capture.wake_chunks(), asyncio.Event()))
        await asyncio.sleep(0)

        for group in (range(0, 4), range(4, 8)):
            capture._ingest(b"".join(_frame(i) for i in group))
            await asyncio.sleep(0)
        await asyncio.wait_for(transport.enter_started.wait(), 1)

        capture._ingest(b"".join(_frame(i) for i in range(8, 12)))
        await asyncio.wait_for(transport.entered.wait(), 1)
        capture._ingest(b"".join(_frame(i) for i in range(12, 16)))

        turn_task = lp._turn_task
        assert turn_task is not None
        turn = await asyncio.wait_for(turn_task, 1)
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

        assert transport.entered_at - transport.enter_started_at >= 0.5
        assert transport.sent == [_frame(i) for i in range(16)]
        assert b"".join(transport.sent) == b"".join(_frame(i) for i in range(16))
        assert turn.transcript == "check what Codex is doing"
        assert turn.tier == "reasoning"
        assert played == [loop_mod.ACK_REASONING]
        required = {
            "wake_detected", "stt_connect_started", "stt_connected",
            "stt_first_partial", "speech_ended_vad", "stt_final",
            "transcript_normalized", "intent_classified", "ack_audible",
        }
        assert required <= turn.stages.keys()
        assert capture.metrics.active_subscribers == 0
        assert isinstance(capture.metrics, AudioMetrics)
        assert capture.metrics.frames_captured == 16
        assert capture.metrics.bytes_captured == 16 * FRAME_BYTES
        assert not [task for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()]

    asyncio.run(run())


def test_wake_only_opens_follow_up_and_only_command_routes(tmp_path, monkeypatch):
    async def run():
        capture = AudioCaptureService(Source())
        detector = Detector()
        first = ScriptedSTT((_final("Friday"),), expected_audio=1)
        follow_up = ScriptedSTT(
            (_partial("check"), _final("check what Codex is doing")),
            expected_audio=1,
        )
        timeout = ScriptedSTT()
        transports = iter((first, follow_up, timeout))
        monkeypatch.setattr(stt, "LocalVAD", lambda: CountVAD(1))
        lp, played = _loop(
            tmp_path, detector, capture, lambda: next(transports),
            conversation_seconds=0.05,
        )
        capture._ingest(_frame(0))
        initial = capture.subscribe_stt()
        task = asyncio.create_task(
            lp.handle_detection(_detection(), subscription=initial)
        )

        await asyncio.wait_for(follow_up.entered.wait(), 1)
        await asyncio.sleep(0.01)  # deliberate pause after wake-only
        capture._ingest(_frame(1))
        wake_turn = await asyncio.wait_for(task, 1)

        assert wake_turn.transcript == ""
        assert wake_turn.tier is None
        assert detector.calls == 0
        assert len(lp.turns) == 2
        command = lp.turns[1]
        assert command.transcript == "check what Codex is doing"
        assert command.tier == "reasoning"
        assert {"stt_connect_started", "stt_connected", "stt_first_partial",
                "speech_ended_vad", "stt_final", "transcript_normalized",
                "intent_classified", "ack_audible"} <= command.stages.keys()
        assert played == [loop_mod.ACK_REASONING]
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (
        ("Friday what is running", "what is running"),
        ("Fridaywhat is running", "what is running"),
        ("Friday stop", "stop"),
    ),
)
def test_single_utterance_wake_prefix_forms_route_once(
    tmp_path, raw, normalized
):
    transport = ScriptedSTT((_final(raw),))
    lp, _ = _loop(tmp_path, Detector(), None, lambda: transport)
    routed: list[str] = []

    async def record(transcript, span, turn):
        routed.append(transcript)

    lp._respond = record
    turn = asyncio.run(lp.handle_detection(_detection()))
    assert turn.transcript == normalized
    assert routed == [normalized]


def test_local_vad_timeout_closes_long_pause(tmp_path):
    async def run():
        capture = AudioCaptureService(Source())
        capture._ingest(_frame(0))
        subscription = capture.subscribe_stt()
        transport = ScriptedSTT(expected_audio=1)
        span = TurnSpan("pending", path=tmp_path / "spans.jsonl")
        started = time.monotonic()
        events = [event async for event in stt.run_utterance(
            subscription, transport, span=span, vad=CountVAD(1), max_wait_ms=20
        )]
        elapsed = time.monotonic() - started
        subscription.close()

        assert events == []
        assert elapsed >= 0.02
        assert span.stages["speech_ended_vad"] <= span.stages["stt_final"]
        assert transport.closed
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


def test_false_wake_and_second_wake_while_listening_are_ignored(tmp_path):
    async def frames():
        yield b"x" * FRAME_BYTES * 4

    async def run():
        capture = AudioCaptureService(Source())
        detector = Detector(wake_on=2)
        lp, _ = _loop(tmp_path, detector, capture, lambda: ScriptedSTT())
        await lp._pump(frames(), asyncio.Event())
        assert detector.calls == 1
        assert lp._turn_task is None

        lp.audio_state = loop_mod.AudioTurnState.LISTENING
        await lp._pump(frames(), asyncio.Event())
        assert detector.calls == 1
        assert lp._turn_task is None
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


@pytest.mark.parametrize("stage", ("connecting", "command"))
def test_mic_preemption_cancels_and_forgets_in_flight_audio(
    tmp_path, monkeypatch, stage
):
    async def run():
        capture = AudioCaptureService(Source())
        capture._ingest(_frame(0))
        detector = Detector()
        transport = ScriptedSTT(block_enter=stage == "connecting")
        monkeypatch.setattr(stt, "LocalVAD", lambda: CountVAD(100))
        lp, _ = _loop(tmp_path, detector, capture, lambda: transport)
        subscription = capture.subscribe_stt()
        task = asyncio.create_task(
            lp.handle_detection(_detection(), subscription=subscription)
        )
        lp._turn_task = task
        ready = transport.enter_started if stage == "connecting" else transport.entered
        await asyncio.wait_for(ready.wait(), 1)

        cleanup = lp._invalidate_turn([])
        assert cleanup is not None
        await asyncio.wait_for(cleanup, 1)
        assert task.cancelled()
        assert lp.audio_state is loop_mod.AudioTurnState.SUSPENDED
        assert lp._stt_subscription is None
        assert capture.snapshot() == ()
        assert capture.metrics.active_subscribers == 0
        assert detector.armed == 1
        await asyncio.sleep(0)
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


def test_connection_failure_cleans_up_and_rearms(tmp_path):
    async def run():
        capture = AudioCaptureService(Source())
        capture._ingest(_frame(0))
        detector = Detector()
        transport = ScriptedSTT(fail_enter=True)
        lp, _ = _loop(tmp_path, detector, capture, lambda: transport)
        turn = await lp.handle_detection(
            _detection(), subscription=capture.subscribe_stt()
        )

        assert "scripted connect failure" in (turn.error or "")
        assert lp.audio_state is loop_mod.AudioTurnState.IDLE
        assert lp._stt_subscription is None
        assert capture.metrics.active_subscribers == 0
        assert detector.armed == 1

    asyncio.run(run())


def test_wake_during_tts_is_ignored_without_stopping_speech(tmp_path):
    async def run():
        capture = AudioCaptureService(Source())
        capture._ingest(_frame(0))
        detector = Detector(wake_on=1)
        lp, _ = _loop(tmp_path, detector, capture, lambda: ScriptedSTT())
        speaker = Speaker()
        async def old_turn():
            await asyncio.Event().wait()

        async def new_turn(detection, *, subscription, span):
            raise AssertionError("wake during TTS must not start a new turn")

        async def wake_frame():
            yield b"x" * FRAME_BYTES * 4

        old = asyncio.create_task(old_turn())
        await asyncio.sleep(0)
        lp._turn_task = old
        lp._speaker = speaker
        lp.audio_state = loop_mod.AudioTurnState.SPEAKING
        lp.handle_detection = new_turn
        await lp._pump(wake_frame(), asyncio.Event())
        assert lp._turn_task is old
        assert not speaker.stopped
        assert not old.done()
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


def test_wake_during_ack_is_ignored(tmp_path):
    async def run():
        capture = AudioCaptureService(Source())
        detector = Detector(wake_on=1)
        ack_started = threading.Event()
        release_ack = threading.Event()

        def play_ack(name, span):
            ack_started.set()
            release_ack.wait(timeout=1)

        lp = loop_mod.VoiceLoop(
            detector=detector,
            capture=capture,
            play_ack=play_ack,
            conversation_seconds=0,
        )
        async def old_turn():
            await lp._play(loop_mod.ACK_REASONING, TurnSpan("pending"))

        async def new_turn(detection, *, subscription, span):
            raise AssertionError("wake during acknowledgement must be ignored")

        async def wake_frame():
            yield b"x" * FRAME_BYTES * 4

        old = asyncio.create_task(old_turn())
        lp._turn_task = old
        await asyncio.wait_for(asyncio.to_thread(ack_started.wait), 1)
        lp.handle_detection = new_turn
        await lp._pump(wake_frame(), asyncio.Event())
        assert lp._turn_task is old
        assert not old.done()
        release_ack.set()
        await asyncio.wait_for(old, 1)
        assert capture.metrics.active_subscribers == 0

    asyncio.run(run())


def test_manager_capture_cancellation_drains_all_race_waiters():
    @contextlib.asynccontextmanager
    async def opener():
        async def frames():
            while True:
                await asyncio.Event().wait()
                yield b"x"

        yield frames()

    async def run():
        manager = AudioResourceManager(
            open_capture=opener, manage_echo_cancel=False
        )
        stop = asyncio.Event()
        capture = asyncio.create_task(anext(manager.capture(stop)))
        await asyncio.sleep(0)
        capture.cancel()
        with pytest.raises(asyncio.CancelledError):
            await capture
        await asyncio.sleep(0)

        assert not [task for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()]

    asyncio.run(run())


def test_tts_cleanup_preserves_outer_turn_cancellation(tmp_path):
    async def text():
        yield "This synthesis is deliberately still running."

    async def run():
        lp = loop_mod.VoiceLoop(
            tts_factory=lambda: tts.FakeSynthesisTransport(
                sentence_delay=10
            ),
            audio_output=Sink(),
            spans_path=tmp_path / "spans.jsonl",
        )
        task = asyncio.create_task(lp._speak_stream(
            text(), TurnSpan("pending"), loop_mod.Turn("pending"), None
        ))
        while lp._speaker is None:
            await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert lp._speaker is None
        assert not [pending for pending in asyncio.all_tasks()
                    if pending is not asyncio.current_task()
                    and not pending.done()]

    asyncio.run(run())
