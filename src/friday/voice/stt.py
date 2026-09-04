"""Deepgram streaming STT with adaptive conversational endpointing.

Consumes AudioCaptureService subscriptions (with the legacy `WakeDetection`
handoff retained for file/tests) and streams them to Deepgram's Flux v2 socket,
while a local webrtcvad-based detector supplies speech activity. Flux EndOfTurn
is authoritative; bounded adaptive deadlines are the fallback when Flux does
not decide. A local pause never closes the audio stream by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional, Protocol

from friday.audio.capture import AudioSubscription
from friday.core import events
from friday.core.spans import TurnSpan, start_turn
from friday.voice.wake import CHUNK_SAMPLES, PREROLL_SECONDS, SAMPLE_RATE, WakeDetection

# FRIDAY's local VAD decides when a command is over. Keep Flux from splitting a
# turn on a thinking pause; CloseStream still flushes EndOfTurn immediately.
DEFAULT_EOT_TIMEOUT_MS = 60_000
DEFAULT_EOT_THRESHOLD = 0.9
WARM_SOCKET_TTL_SECONDS = 120.0

VAD_FRAME_MS = 10
VAD_FRAME_BYTES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 320 bytes = 160 samples
VAD_AGGRESSIVENESS = 2
FLUSH_WAIT_MS = 200


@dataclass(frozen=True)
class EndpointConfig:
    vad_pause_ms: int = 400
    fast_ms: int = 650
    patient_ms: int = 1800
    max_ms: int = 2500

    def __post_init__(self) -> None:
        if not (0 < self.vad_pause_ms <= self.fast_ms < self.patient_ms <= self.max_ms):
            raise ValueError(
                "invalid endpoint configuration: require 0 < VAD pause <= fast < patient <= max"
            )

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        names = {
            "vad_pause_ms": "FRIDAY_VAD_PAUSE_MS",
            "fast_ms": "FRIDAY_ENDPOINT_FAST_MS",
            "patient_ms": "FRIDAY_ENDPOINT_PATIENT_MS",
            "max_ms": "FRIDAY_ENDPOINT_MAX_MS",
        }
        values = {}
        for field, name in names.items():
            raw = os.environ.get(name)
            if raw is not None:
                try:
                    values[field] = int(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid endpoint configuration: {name} must be an integer"
                    ) from exc
        return cls(**values)


class EndpointState(Enum):
    LISTENING = "listening"
    SPEAKING = "speaking"
    POSSIBLE_END = "possible_end"
    WAITING_FOR_EOT = "waiting_for_eot"
    FINALIZED = "finalized"


class SpeechSignal(Enum):
    NONE = "none"
    STARTED = "started"
    PAUSE = "pause"
    RESUMED = "resumed"


@dataclass
class EndpointStats:
    resumed_pauses: int = 0
    longest_pause_ms: int = 0
    utterance_started_at: Optional[float] = None
    final_speech_at: Optional[float] = None
    finalized_at: Optional[float] = None


_CONTINUATION_WORDS = frozenset(
    {
        "and",
        "or",
        "but",
        "because",
        "so",
        "then",
        "to",
        "with",
        "if",
        "that",
    }
)
_FAST_REFLEXES = frozenset({"stop", "cancel", "yes", "no"})
_QUESTION_STARTS = frozenset(
    {
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "how",
        "is",
        "should",
        "was",
        "were",
        "what",
        "what's",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "would",
    }
)


class EndpointController:
    """Deterministic fusion of local speech activity and Flux turn events."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        stats: Optional[EndpointStats] = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.state = EndpointState.LISTENING
        self.transcript = ""
        self.deadline: Optional[float] = None
        self.absolute_deadline: Optional[float] = None
        self.pause_started_at: Optional[float] = None
        self.finalized_reason: Optional[str] = None
        self.stats = stats or EndpointStats()

    def on_speech(self, signal: SpeechSignal, now: Optional[float] = None) -> Optional[str]:
        now = self.clock() if now is None else now
        if self.state is EndpointState.FINALIZED or signal is SpeechSignal.NONE:
            return self.finalized_reason
        if signal is SpeechSignal.STARTED:
            self.stats.utterance_started_at = self.stats.utterance_started_at or now
            self.state = EndpointState.SPEAKING
            events.debug("turn", "speech-start")
        elif signal is SpeechSignal.PAUSE:
            if self.pause_started_at is not None:
                return self.finalized_reason
            self.pause_started_at = now - self.config.vad_pause_ms / 1000
            self.stats.final_speech_at = self.pause_started_at
            grace_ms = self.config.fast_ms if self._fast_candidate() else self.config.patient_ms
            self.deadline = self.pause_started_at + grace_ms / 1000
            self.absolute_deadline = self.pause_started_at + self.config.max_ms / 1000
            self.state = EndpointState.POSSIBLE_END
            events.debug(
                "turn",
                "possible-end",
                silence_ms=self.config.vad_pause_ms,
                grace_ms=grace_ms,
            )
        elif signal is SpeechSignal.RESUMED:
            if self.pause_started_at is not None:
                pause_ms = round((now - self.pause_started_at) * 1000)
                self.stats.resumed_pauses += 1
                self.stats.longest_pause_ms = max(self.stats.longest_pause_ms, pause_ms)
                events.debug("turn", "speech-resumed", pause_ms=pause_ms)
            self.pause_started_at = None
            self.deadline = None
            self.absolute_deadline = None
            self.state = EndpointState.SPEAKING
        return self.finalized_reason

    def on_flux(self, event: "TranscriptEvent", now: Optional[float] = None) -> Optional[str]:
        now = self.clock() if now is None else now
        if event.text.strip():
            self.transcript = event.text.strip()
            if self.pause_started_at is not None and self._fast_candidate():
                self.deadline = min(
                    self.deadline or float("inf"),
                    self.pause_started_at + self.config.fast_ms / 1000,
                )
        if self.state is EndpointState.FINALIZED:
            return None
        if event.turn_event == "StartOfTurn":
            signal = (
                SpeechSignal.RESUMED if self.pause_started_at is not None else SpeechSignal.STARTED
            )
            return self.on_speech(signal, now)
        if event.turn_event == "TurnResumed":
            return self.on_speech(SpeechSignal.RESUMED, now)
        if event.turn_event == "EagerEndOfTurn" and self.pause_started_at is not None:
            self.state = EndpointState.WAITING_FOR_EOT
        if event.turn_event == "EndOfTurn" or event.speech_final:
            return self.finalize("flux_eot", now)
        return None

    def on_timeout(self, now: Optional[float] = None) -> Optional[str]:
        now = self.clock() if now is None else now
        if self.state is EndpointState.FINALIZED or self.deadline is None:
            return self.finalized_reason
        if self.absolute_deadline is not None and now >= self.absolute_deadline:
            return self.finalize("absolute_timeout", now)
        if now >= self.deadline:
            reason = "fast_timeout" if self._fast_candidate() else "patient_timeout"
            return self.finalize(reason, now)
        return None

    def finalize(self, reason: str, now: Optional[float] = None) -> str:
        if self.finalized_reason is not None:
            return self.finalized_reason
        now = self.clock() if now is None else now
        self.finalized_reason = reason
        self.stats.finalized_at = now
        self.state = EndpointState.FINALIZED
        self.deadline = None
        self.absolute_deadline = None
        endpoint_ms = (
            round((now - self.stats.final_speech_at) * 1000)
            if self.stats.final_speech_at is not None
            else None
        )
        utterance_ms = (
            round((now - self.stats.utterance_started_at) * 1000)
            if self.stats.utterance_started_at is not None
            else None
        )
        events.emit(
            "turn",
            "finalized",
            reason=reason,
            endpoint_ms=endpoint_ms,
            utterance_ms=utterance_ms,
            resumed=self.stats.resumed_pauses,
            longest_pause_ms=self.stats.longest_pause_ms,
        )
        return reason

    def _fast_candidate(self) -> bool:
        words = re.findall(r"[\w']+", self.transcript.lower())
        if words[:2] == ["hey", "friday"]:
            words = words[2:]
        elif words[:1] == ["friday"]:
            words = words[1:]
        if not words or len(words) > 8 or words[-1] in _CONTINUATION_WORDS:
            return False
        return len(words) == 1 and words[0] in _FAST_REFLEXES or words[0] in _QUESTION_STARTS


@dataclass
class TranscriptEvent:
    """One transcript update from the STT transport."""

    text: str
    is_final: bool
    speech_final: bool = False
    utterance_end: bool = False
    turn_event: Optional[str] = None
    eot_confidence: Optional[float] = None
    hard_stop: bool = False


def is_hard_stop_command(text: str) -> bool:
    """Match only the isolated imperative, never a containing sentence."""
    words = re.findall(r"[a-z']+", text.lower())
    return words in (["friday", "stop"], ["hey", "friday", "stop"])


class HardStopMatcher:
    """Require a Flux turn signal or two identical exact streaming updates."""

    def __init__(self) -> None:
        self._exact_updates = 0

    def feed(self, event: TranscriptEvent) -> bool:
        if not is_hard_stop_command(event.text):
            self._exact_updates = 0
            return False
        if event.turn_event in {"EagerEndOfTurn", "EndOfTurn"} or event.speech_final:
            return True
        if event.turn_event in {None, "Update"}:
            self._exact_updates += 1
        return self._exact_updates >= 2


class Transport(Protocol):
    """Minimal async streaming STT transport, implemented by DeepgramTransport
    (real) and a fake in tests. Injected, never branched on internally."""

    async def __aenter__(self) -> "Transport": ...

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]: ...

    async def send_media(self, chunk: bytes) -> None: ...

    async def send_close_stream(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]: ...


class DeepgramTransport:
    """Real transport: wraps Deepgram Flux via the v2 streaming endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "flux-general-multi",
        sample_rate: int = SAMPLE_RATE,
        eot_timeout_ms: int = DEFAULT_EOT_TIMEOUT_MS,
        eot_threshold: float = DEFAULT_EOT_THRESHOLD,
        extra_config: Optional[dict] = None,
    ) -> None:
        from deepgram import AsyncDeepgramClient

        self._client = AsyncDeepgramClient(api_key=api_key)
        self.config: dict = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "eot_timeout_ms": eot_timeout_ms,
            "eot_threshold": eot_threshold,
            **(extra_config or {}),
        }
        self._connect_cm = None
        self._connection = None
        self._listen_task: Optional[asyncio.Task] = None
        # None is the end-of-stream sentinel: Deepgram's socket closing (or
        # send_close_stream being acknowledged) has to terminate iteration,
        # otherwise a consumer awaits a queue that will never fill again.
        self._queue: "asyncio.Queue[Optional[TranscriptEvent]]" = asyncio.Queue()

    async def __aenter__(self) -> "DeepgramTransport":
        from deepgram.core.events import EventType

        self._connect_cm = self._client.listen.v2.connect(**self.config)
        self._connection = await self._connect_cm.__aenter__()
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._listen_task = asyncio.create_task(self._run_listener())
        return self

    async def _run_listener(self) -> None:
        """Pump the socket, then close iteration however it ends."""
        try:
            await self._connection.start_listening()
        finally:
            self._queue.put_nowait(None)

    def _on_message(self, message: Any) -> None:
        msg_type = getattr(message, "type", None)
        if msg_type == "TurnInfo":
            end_of_turn = message.event == "EndOfTurn"
            self._queue.put_nowait(
                TranscriptEvent(
                    text=message.transcript,
                    is_final=end_of_turn,
                    speech_final=end_of_turn,
                    turn_event=message.event,
                    eot_confidence=getattr(message, "end_of_turn_confidence", None),
                )
            )

    async def send_media(self, chunk: bytes) -> None:
        await self._connection.send_media(chunk)

    async def send_close_stream(self) -> None:
        await self._connection.send_close_stream()

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        if self._listen_task is not None:
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task
        if self._connect_cm is not None:
            await self._connect_cm.__aexit__(*exc_info)
        return None

    @property
    def connected(self) -> bool:
        """Cheap liveness check for a parked, pre-warmed connection."""
        return self._connection is not None and (
            self._listen_task is not None and not self._listen_task.done()
        )


class FluxTransportPool:
    """Keep one exact-config Flux socket warm between sequential voice turns."""

    def __init__(
        self,
        api_key: str,
        *,
        ttl_seconds: float = WARM_SOCKET_TTL_SECONDS,
        transport_factory: Callable[..., DeepgramTransport] = DeepgramTransport,
        **config: Any,
    ) -> None:
        self._api_key = api_key
        self._ttl = ttl_seconds
        self._transport_factory = transport_factory
        self._config = config
        self._warm_task: Optional[asyncio.Task] = None

    def prewarm(self) -> None:
        if self._warm_task is None:
            self._warm_task = asyncio.create_task(self._open())

    async def _open(self) -> tuple[DeepgramTransport, float]:
        transport = self._transport_factory(self._api_key, **self._config)
        await transport.__aenter__()
        return transport, time.monotonic()

    def __call__(self) -> "_FluxLease":
        return _FluxLease(self)

    async def _acquire(self) -> DeepgramTransport:
        task, self._warm_task = self._warm_task, None
        if task is not None:
            try:
                transport, opened_at = await task
                if transport.connected and time.monotonic() - opened_at <= self._ttl:
                    return transport
                await transport.__aexit__(None, None, None)
            except Exception as exc:
                events.debug("stt-connection", "warm-reuse-failed", error=repr(exc))
        transport = self._transport_factory(self._api_key, **self._config)
        return await transport.__aenter__()

    async def close(self) -> None:
        task, self._warm_task = self._warm_task, None
        if task is None:
            return
        try:
            transport, _ = await task
        except (Exception, asyncio.CancelledError):
            return
        await transport.__aexit__(None, None, None)


class _FluxLease:
    def __init__(self, pool: FluxTransportPool) -> None:
        self._pool = pool
        self._transport: Optional[DeepgramTransport] = None

    async def __aenter__(self) -> DeepgramTransport:
        self._transport = await self._pool._acquire()
        return self._transport

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        assert self._transport is not None
        try:
            return await self._transport.__aexit__(*exc_info)
        finally:
            self._pool.prewarm()


class LocalVAD:
    """webrtcvad speech activity detector that preserves pause/resume state."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS, silence_ms: int = 400) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames_needed = max(1, silence_ms // VAD_FRAME_MS)
        self._silence_run = 0
        self._heard_speech = False
        self._paused = False

    def feed(self, chunk: bytes) -> SpeechSignal:
        """Feed one 80ms chunk (split into 10ms sub-frames for webrtcvad).

        Returns only state transitions; quiet frames between transitions are NONE.
        """
        result = SpeechSignal.NONE
        for i in range(0, len(chunk) - VAD_FRAME_BYTES + 1, VAD_FRAME_BYTES):
            frame = chunk[i : i + VAD_FRAME_BYTES]
            if self._vad.is_speech(frame, SAMPLE_RATE):
                if not self._heard_speech:
                    result = SpeechSignal.STARTED
                elif self._paused:
                    result = SpeechSignal.RESUMED
                self._heard_speech = True
                self._paused = False
                self._silence_run = 0
            elif self._heard_speech:
                self._silence_run += 1
                if not self._paused and self._silence_run >= self._silence_frames_needed:
                    self._paused = True
                    result = SpeechSignal.PAUSE
        return result


async def run_utterance(
    source: "WakeDetection | AudioSubscription",
    transport: Optional[Transport],
    *,
    span: Optional[TurnSpan] = None,
    vad: Optional[LocalVAD] = None,
    endpoint_config: Optional[EndpointConfig] = None,
    endpoint_stats: Optional[EndpointStats] = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[TranscriptEvent]:
    """Stream one semantic turn, preserving audio through thinking pauses.

    Local VAD changes endpoint state but never closes the stream. Flux EndOfTurn
    is authoritative; adaptive local deadlines are bounded fallbacks.

    `transport`, when given, must already be entered (`async with`); this
    function does not own its lifecycle. Passing an un-entered transport raises
    out of the iterator rather than yielding an empty turn.
    """
    config = endpoint_config or EndpointConfig.from_env()
    vad = vad or LocalVAD(silence_ms=config.vad_pause_ms)
    endpoint = EndpointController(config, clock=clock, stats=endpoint_stats)
    hard_stop_matcher = HardStopMatcher()
    out_queue: "asyncio.Queue[Any]" = asyncio.Queue()
    audio_done = asyncio.Event()
    got_final = asyncio.Event()
    finalized = asyncio.Event()
    state_changed = asyncio.Event()
    pending_audio = False
    flushed = False
    sentinel = object()

    async def audio_source() -> AsyncIterator[bytes]:
        if isinstance(source, AudioSubscription):
            async for frame in source:
                if frame.pcm:
                    yield frame.pcm
            return
        # Never emit a zero-length chunk: Deepgram reads an empty binary frame
        # as an end-of-stream signal and closes the socket, which looks exactly
        # like a mid-turn network failure. An empty preroll is legitimate (a
        # turn opened without buffered audio), so it must be skipped, not sent.
        if source.preroll:
            yield source.preroll
        while True:
            chunk = await source.live.get()
            if chunk is None:
                return
            if chunk:
                yield chunk

    async def pump() -> None:
        nonlocal pending_audio
        try:
            async for chunk in audio_source():
                raw_signal = vad.feed(chunk)
                signal = (
                    raw_signal
                    if isinstance(raw_signal, SpeechSignal)
                    else (SpeechSignal.PAUSE if raw_signal else SpeechSignal.NONE)
                )
                if transport is not None:
                    await transport.send_media(chunk)
                    pending_audio = True
                if signal is not SpeechSignal.NONE:
                    endpoint.on_speech(signal, clock())
                    state_changed.set()
                if signal is SpeechSignal.PAUSE:
                    if span is not None:
                        span.mark("speech_ended_vad")
                elif signal is SpeechSignal.RESUMED and span is not None:
                    span.mark("speech_resumed")
                if finalized.is_set():
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Do not let a dead pump degrade into a silent empty turn: without
            # this the consumer just sees no transcript and FRIDAY appears to
            # have stopped hearing, with nothing in the logs to say why.
            await out_queue.put(exc)
        finally:
            audio_done.set()

    async def recv() -> None:
        if transport is None:
            return
        try:
            async for event in transport:
                if hard_stop_matcher.feed(event):
                    event.hard_stop = True
                await out_queue.put(event)
                if event.hard_stop:
                    endpoint.finalize("hard_stop", clock())
                    finalized.set()
                    return
                reason = endpoint.on_flux(event, clock())
                state_changed.set()
                if event.speech_final or event.utterance_end:
                    got_final.set()
                if reason is not None:
                    finalized.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await out_queue.put(exc)

    async def endpoint_timer() -> None:
        while not finalized.is_set():
            if audio_done.is_set():
                if span is not None and "speech_ended_vad" not in span.stages:
                    span.mark("speech_ended_vad")
                endpoint.finalize("audio_exhausted", clock())
                finalized.set()
                return
            state_changed.clear()
            deadline = endpoint.deadline
            if deadline is None:
                waiters = [
                    asyncio.create_task(state_changed.wait()),
                    asyncio.create_task(audio_done.wait()),
                ]
                try:
                    await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for waiter in waiters:
                        waiter.cancel()
                continue
            delay = max(0.0, deadline - clock())
            try:
                await asyncio.wait_for(state_changed.wait(), timeout=delay)
            except TimeoutError:
                if endpoint.on_timeout(clock()) is not None:
                    finalized.set()

    async def closer() -> None:
        nonlocal flushed
        await finalized.wait()
        if transport is not None and pending_audio:
            await transport.send_close_stream()
            flushed = True
            if endpoint.finalized_reason != "hard_stop":
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(got_final.wait(), timeout=FLUSH_WAIT_MS / 1000)
        if span is not None:
            span.mark("stt_final")
        await out_queue.put(sentinel)

    pump_task = asyncio.create_task(pump())
    recv_task = asyncio.create_task(recv())
    timer_task = asyncio.create_task(endpoint_timer())
    closer_task = asyncio.create_task(closer())

    try:
        while True:
            item = await out_queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        for task in (pump_task, recv_task, timer_task, closer_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(pump_task, recv_task, timer_task, closer_task, return_exceptions=True)
        if transport is not None and not flushed:
            with contextlib.suppress(Exception):
                await transport.send_close_stream()


def _require_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        print(
            "error: DEEPGRAM_API_KEY is not set. Export DEEPGRAM_API_KEY=<your key> and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key


def _detection_from_wav(path: str, turn_id: str = "cli") -> WakeDetection:
    """Build a WakeDetection-shaped handoff from a WAV file for --file testing:
    the first PREROLL_SECONDS becomes `preroll`, the rest is queued as `live`."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16kHz mono 16-bit PCM")
        raw = w.readframes(w.getnframes())
    frame_bytes = CHUNK_SAMPLES * 2
    chunks = [raw[i : i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
    preroll_chunks = max(1, int(PREROLL_SECONDS * SAMPLE_RATE) // CHUNK_SAMPLES)
    preroll = b"".join(chunks[:preroll_chunks])
    live: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    for chunk in chunks[preroll_chunks:]:
        live.put_nowait(chunk)
    live.put_nowait(None)
    return WakeDetection(
        model="cli-file",
        score=1.0,
        timestamp=time.time(),
        turn_id=turn_id,
        preroll=preroll,
        preroll_samples=len(preroll) // 2,
        preroll_seconds=(len(preroll) // 2) / SAMPLE_RATE,
        live=live,
    )


async def _run_file(path: str, eot_timeout_ms: int) -> None:
    api_key = _require_api_key()
    detection = _detection_from_wav(path)
    span = start_turn("reflex", turn_id=detection.turn_id)
    async with DeepgramTransport(api_key, eot_timeout_ms=eot_timeout_ms) as transport:
        async for event in run_utterance(detection, transport, span=span):
            tag = "FINAL" if event.is_final else "interim"
            print(f"[{tag}] {event.text}")
    span.write()


async def _run_live(eot_timeout_ms: int) -> None:
    api_key = _require_api_key()
    from friday.voice.wake import WakeWordDetector, capture_loop

    detector = WakeWordDetector()

    async def _handle(detection: WakeDetection) -> None:
        span = start_turn("reflex", turn_id=detection.turn_id)
        async with DeepgramTransport(api_key, eot_timeout_ms=eot_timeout_ms) as transport:
            async for event in run_utterance(detection, transport, span=span):
                tag = "FINAL" if event.is_final else "interim"
                print(f"[{tag}] {event.text}")
        detector.end_handoff()
        span.write()

    def _on_detection(detection: WakeDetection) -> None:
        asyncio.create_task(_handle(detection))

    await capture_loop(detector, _on_detection)


async def endpoint_probe() -> int:
    """Capture and transcribe one turn with endpoint diagnostics, never Claude."""
    from friday.audio.capture import AudioCaptureService
    from friday.audio.manager import AudioResourceManager

    config = EndpointConfig.from_env()
    stats = EndpointStats()
    manager = AudioResourceManager()
    capture = AudioCaptureService(manager)
    stop = asyncio.Event()
    await manager.start()
    capture_task = asyncio.create_task(capture.run(stop))
    subscription = capture.subscribe_live()
    transcript = ""
    print("Speak one natural request now. Ctrl-C cancels.\n")
    try:
        async with DeepgramTransport(_require_api_key()) as transport:
            async for event in run_utterance(
                subscription,
                transport,
                endpoint_config=config,
                endpoint_stats=stats,
            ):
                if event.text.strip():
                    transcript = event.text.strip()
    finally:
        subscription.close()
        stop.set()
        if not capture_task.done():
            capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)
        await manager.stop()

    endpoint_ms = (
        round((stats.finalized_at - stats.final_speech_at) * 1000)
        if stats.finalized_at is not None and stats.final_speech_at is not None
        else None
    )
    print(f"\ntranscript:\n{transcript!r}\n")
    print(f"endpoint latency: {endpoint_ms if endpoint_ms is not None else 'unknown'} ms")
    print(f"pauses survived: {stats.resumed_pauses}")
    print(f"longest pause: {stats.longest_pause_ms} ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.voice.stt")
    parser.add_argument("--file", help="run a 16kHz mono 16-bit WAV through the full STT path")
    parser.add_argument(
        "--live",
        action="store_true",
        help="run live mic capture through the full STT path",
    )
    parser.add_argument("--eot-timeout-ms", type=int, default=DEFAULT_EOT_TIMEOUT_MS)
    args = parser.parse_args()

    if args.file:
        asyncio.run(_run_file(args.file, args.eot_timeout_ms))
        return 0
    if args.live:
        asyncio.run(_run_live(args.eot_timeout_ms))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
