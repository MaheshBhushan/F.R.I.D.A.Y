"""Deepgram streaming STT: replay wake.py's pre-roll + live audio, surface interim
and final transcripts, and close the turn on local VAD independent of the network.

Consumes AudioCaptureService subscriptions (with the legacy `WakeDetection`
handoff retained for file/tests) and streams them to Deepgram's v1 socket with
interim results on, while a local webrtcvad-based detector marks speech-end from
raw audio alone. The turn closes on agreement between local VAD and Deepgram's own
endpointing/UtteranceEnd signal, capped at MAX_WAIT_MS so a flaky connection can
never stall the turn indefinitely.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
import wave
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Protocol

from friday.audio.capture import AudioSubscription
from friday.core.spans import TurnSpan, start_turn
from friday.voice.wake import CHUNK_SAMPLES, PREROLL_SECONDS, SAMPLE_RATE, WakeDetection

# Deepgram's SDK default is 10ms of trailing silence before it finalizes a result -
# fast, but eager enough to finalize on a mid-sentence breath. We raise it to 100ms
# so Deepgram-side finalization isn't the thing flapping on every short pause; the
# real turn boundary is local VAD + MAX_WAIT_MS below, not Deepgram's own timer, so
# this only affects how eagerly is_final/speech_final events arrive, not turn-taking.
DEFAULT_ENDPOINTING_MS = 100

# Max time to wait, after local VAD calls speech-end, for Deepgram to agree before
# closing the turn on best-available evidence anyway.
MAX_WAIT_MS = 700

VAD_FRAME_MS = 10
VAD_FRAME_BYTES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 320 bytes = 160 samples
VAD_SILENCE_MS = 400  # trailing silence required before local VAD calls speech-end
VAD_AGGRESSIVENESS = 2


@dataclass
class TranscriptEvent:
    """One transcript update from the STT transport."""

    text: str
    is_final: bool
    speech_final: bool = False
    utterance_end: bool = False


class Transport(Protocol):
    """Minimal async streaming STT transport, implemented by DeepgramTransport
    (real) and a fake in tests. Injected, never branched on internally."""

    async def __aenter__(self) -> "Transport": ...

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]: ...

    async def send_media(self, chunk: bytes) -> None: ...

    async def send_close_stream(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]: ...


class DeepgramTransport:
    """Real transport: wraps deepgram-sdk's AsyncDeepgramClient.listen.v1.connect."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        sample_rate: int = SAMPLE_RATE,
        endpointing: int = DEFAULT_ENDPOINTING_MS,
        extra_config: Optional[dict] = None,
    ) -> None:
        from deepgram import AsyncDeepgramClient

        self._client = AsyncDeepgramClient(api_key=api_key)
        self.config: dict = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": True,
            "endpointing": endpointing,
            "vad_events": True,
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

        self._connect_cm = self._client.listen.v1.connect(**self.config)
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
        if msg_type == "Results":
            alt = message.channel.alternatives[0]
            self._queue.put_nowait(
                TranscriptEvent(
                    text=alt.transcript,
                    is_final=bool(message.is_final),
                    speech_final=bool(message.speech_final),
                )
            )
        elif msg_type == "UtteranceEnd":
            self._queue.put_nowait(TranscriptEvent(text="", is_final=True, utterance_end=True))

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


class LocalVAD:
    """webrtcvad-based speech-end detector, independent of any STT transport."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS, silence_ms: int = VAD_SILENCE_MS) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames_needed = max(1, silence_ms // VAD_FRAME_MS)
        self._silence_run = 0
        self._heard_speech = False

    def feed(self, chunk: bytes) -> bool:
        """Feed one 80ms chunk (split into 10ms sub-frames for webrtcvad).

        Returns True the instant trailing silence >= silence_ms is observed
        after speech has been heard.
        """
        for i in range(0, len(chunk) - VAD_FRAME_BYTES + 1, VAD_FRAME_BYTES):
            frame = chunk[i : i + VAD_FRAME_BYTES]
            if self._vad.is_speech(frame, SAMPLE_RATE):
                self._heard_speech = True
                self._silence_run = 0
            elif self._heard_speech:
                self._silence_run += 1
                if self._silence_run >= self._silence_frames_needed:
                    return True
        return False


async def run_utterance(
    source: "WakeDetection | AudioSubscription",
    transport: Optional[Transport],
    *,
    span: Optional[TurnSpan] = None,
    vad: Optional[LocalVAD] = None,
    max_wait_ms: int = MAX_WAIT_MS,
) -> AsyncIterator[TranscriptEvent]:
    """Replay preroll + live audio (gapless) into `transport`, yielding transcript
    events as they arrive. Local VAD runs on the same audio regardless of whether
    a transport is present, marking `speech_ended_vad`. The turn closes (marking
    `stt_final`) on agreement between local VAD and the transport's final/
    speech_final/utterance_end signal, or after `max_wait_ms` past VAD speech-end,
    whichever comes first.

    `transport`, when given, must already be entered (`async with`); this
    function does not own its lifecycle. Passing an un-entered transport raises
    out of the iterator rather than yielding an empty turn.
    """
    vad = vad or LocalVAD()
    out_queue: "asyncio.Queue[Any]" = asyncio.Queue()
    vad_ended = asyncio.Event()
    audio_done = asyncio.Event()
    got_final = asyncio.Event()
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
        try:
            async for chunk in audio_source():
                if not vad_ended.is_set() and vad.feed(chunk):
                    vad_ended.set()
                    if span is not None:
                        span.mark("speech_ended_vad")
                if transport is not None:
                    await transport.send_media(chunk)
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
                await out_queue.put(event)
                # `is_final` only finalizes one transcript segment; it is not
                # an end-of-turn signal. Deepgram may emit an empty/final wake
                # segment before the command that follows it.
                if event.speech_final or event.utterance_end:
                    got_final.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await out_queue.put(exc)

    async def closer() -> None:
        # Wait for local VAD to call speech-end, but exhausted audio is itself
        # a definitive speech-end: audio that stops without a trailing silence
        # window (a clipped buffer, a file-fed turn, a closed mic stream) would
        # otherwise leave this waiting forever and deadlock the voice loop.
        waiters = [asyncio.create_task(vad_ended.wait()),
                   asyncio.create_task(audio_done.wait())]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()
        if not vad_ended.is_set():
            vad_ended.set()
            if span is not None and "speech_ended_vad" not in span.stages:
                span.mark("speech_ended_vad")
        if transport is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(got_final.wait(), timeout=max_wait_ms / 1000)
        if span is not None:
            span.mark("stt_final")
        await out_queue.put(sentinel)

    pump_task = asyncio.create_task(pump())
    recv_task = asyncio.create_task(recv())
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
        for task in (pump_task, recv_task, closer_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(pump_task, recv_task, closer_task, return_exceptions=True)
        if transport is not None:
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


async def _run_file(path: str, endpointing: int) -> None:
    api_key = _require_api_key()
    detection = _detection_from_wav(path)
    span = start_turn("reflex", turn_id=detection.turn_id)
    async with DeepgramTransport(api_key, endpointing=endpointing) as transport:
        async for event in run_utterance(detection, transport, span=span):
            tag = "FINAL" if event.is_final else "interim"
            print(f"[{tag}] {event.text}")
    span.write()


async def _run_live(endpointing: int) -> None:
    api_key = _require_api_key()
    from friday.voice.wake import WakeWordDetector, capture_loop

    detector = WakeWordDetector()

    async def _handle(detection: WakeDetection) -> None:
        span = start_turn("reflex", turn_id=detection.turn_id)
        async with DeepgramTransport(api_key, endpointing=endpointing) as transport:
            async for event in run_utterance(detection, transport, span=span):
                tag = "FINAL" if event.is_final else "interim"
                print(f"[{tag}] {event.text}")
        detector.end_handoff()
        span.write()

    def _on_detection(detection: WakeDetection) -> None:
        asyncio.create_task(_handle(detection))

    await capture_loop(detector, _on_detection)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.voice.stt")
    parser.add_argument("--file", help="run a 16kHz mono 16-bit WAV through the full STT path")
    parser.add_argument("--live", action="store_true", help="run live mic capture through the full STT path")
    parser.add_argument("--endpointing", type=int, default=DEFAULT_ENDPOINTING_MS)
    args = parser.parse_args()

    if args.file:
        asyncio.run(_run_file(args.file, args.endpointing))
        return 0
    if args.live:
        asyncio.run(_run_live(args.endpointing))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
