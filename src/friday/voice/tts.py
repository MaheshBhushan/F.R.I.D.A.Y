"""Sentence-streamed TTS with hard-preempt barge-in.

Consumes an async text/token stream from the LLM, splits it into sentences as
they complete, and starts synthesizing + playing the first sentence while
later tokens are still arriving (see `TTSSpeaker.speak`). `TTSSpeaker.stop()`
is a hard preempt: it drains the queued audio, cancels the in-flight
synthesis, and cancels the upstream text stream itself -- not a graceful
wind-down.

Real synthesis targets Deepgram's Aura streaming TTS via `deepgram-sdk`'s
`AsyncDeepgramClient.speak.v1.connect(...)` websocket (verified against the
installed 7.7.0 package source: `deepgram/speak/v1/client.py` and
`socket_client.py`), mirroring how `stt.py` uses
`AsyncDeepgramClient.listen.v1.connect(...)`. Transport is injected via the
same Protocol pattern `stt.py` uses for STT: `DeepgramSpeakTransport` (real)
and `FakeSynthesisTransport` (tests), never branched on internally.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
from typing import Any, AsyncIterator, Optional, Protocol

from friday.core.spans import TurnSpan
from friday.voice import indicator

SAMPLE_RATE = 16000  # linear16 mono, matches the mic path's sample rate
CHUNK_MS = 60  # within the 40-80ms band that makes a fast preempt possible
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 16-bit samples

# Sentence/clause boundary: greedily consume non-terminator text up to and
# including a run of .!? , then trailing whitespace (or end of buffer). This
# is intentionally approximate (it will split "3.14" too) -- good enough for
# "start speaking the first sentence", not a real sentence tokenizer.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)")

_DEFAULT_INTERRUPT_PHRASES = ("stop", "wait")


def split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split `buffer` into complete sentences plus a trailing remainder that
    has no terminator yet. Sentences are stripped; the remainder is not."""
    sentences: list[str] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(buffer):
        text = match.group().strip()
        if text:
            sentences.append(text)
        pos = match.end()
    return sentences, buffer[pos:]


class SynthesisTransport(Protocol):
    """Minimal async streaming TTS transport, implemented by
    DeepgramSpeakTransport (real) and FakeSynthesisTransport (tests).
    Injected, never branched on internally."""

    async def __aenter__(self) -> "SynthesisTransport": ...

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]: ...

    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class DeepgramSpeakTransport:
    """Real transport: wraps deepgram-sdk's AsyncDeepgramClient.speak.v1.connect
    (a persistent websocket; each `synthesize()` call sends one Speak + Flush
    and yields raw PCM bytes until the matching Flushed message)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "aura-2-asteria-en",
        sample_rate: int = SAMPLE_RATE,
        encoding: str = "linear16",
        extra_config: Optional[dict] = None,
    ) -> None:
        from deepgram import AsyncDeepgramClient

        self._client = AsyncDeepgramClient(api_key=api_key)
        self.config: dict = {
            "model": model,
            "encoding": encoding,
            "sample_rate": str(sample_rate),
            **(extra_config or {}),
        }
        self._connect_cm = None
        self._connection = None

    async def __aenter__(self) -> "DeepgramSpeakTransport":
        self._connect_cm = self._client.speak.v1.connect(**self.config)
        self._connection = await self._connect_cm.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        if self._connect_cm is not None:
            await self._connect_cm.__aexit__(*exc_info)
        return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        from deepgram.speak.v1.types.speak_v1flushed import SpeakV1Flushed
        from deepgram.speak.v1.types.speak_v1text import SpeakV1Text

        await self._connection.send_text(SpeakV1Text(text=text))
        await self._connection.send_flush()
        async for message in self._connection:
            if isinstance(message, bytes):
                yield message
            elif isinstance(message, SpeakV1Flushed):
                return


class FakeSynthesisTransport:
    """Test double: after a configurable per-sentence delay, yields synthetic
    silence PCM sized to `bytes_per_sentence`. Injected the same way
    DeepgramSpeakTransport is."""

    def __init__(
        self,
        *,
        sentence_delay: float = 0.0,
        bytes_per_sentence: int = SAMPLE_RATE,  # ~0.5s of int16 mono audio
    ) -> None:
        self.sentence_delay = sentence_delay
        self.bytes_per_sentence = bytes_per_sentence
        self.synthesized: list[str] = []

    async def __aenter__(self) -> "FakeSynthesisTransport":
        return self

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized.append(text)
        if self.sentence_delay:
            await asyncio.sleep(self.sentence_delay)
        yield b"\x00" * self.bytes_per_sentence


class AudioOutput(Protocol):
    """Minimal sync audio sink; writes happen off the event loop thread."""

    def write(self, chunk: bytes) -> None: ...

    def close(self) -> None: ...


class SoundDeviceOutput:
    """Real output: a raw int16 mono stream via `sounddevice`."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        import sounddevice as sd

        from friday.voice import devices

        devices.apply()
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=devices.output_device(),
        )
        self._stream.start()

    def write(self, chunk: bytes) -> None:
        self._stream.write(chunk)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.stop()
            self._stream.close()


class _Rechunker:
    """Repacks arbitrary-sized PCM writes into fixed `chunk_bytes` pieces so
    the playback queue never holds anything bigger than one small chunk --
    that's what makes a fast flush possible. The final partial piece (if any)
    is zero-padded to keep every chunk the same, provable duration."""

    def __init__(self, chunk_bytes: int) -> None:
        self._chunk_bytes = chunk_bytes
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        chunks = []
        while len(self._buf) >= self._chunk_bytes:
            chunks.append(bytes(self._buf[: self._chunk_bytes]))
            del self._buf[: self._chunk_bytes]
        return chunks

    def flush(self) -> Optional[bytes]:
        if not self._buf:
            return None
        pad = self._chunk_bytes - len(self._buf)
        chunk = bytes(self._buf) + b"\x00" * pad
        self._buf.clear()
        return chunk


class MicGate:
    """Gates mic transcripts while TTS is speaking, admitting only
    high-confidence interrupt phrases. Not full-duplex barge-in: while not
    speaking, everything is admitted."""

    def __init__(self, interrupt_phrases: tuple[str, ...] = _DEFAULT_INTERRUPT_PHRASES) -> None:
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in interrupt_phrases) + r")\b", re.IGNORECASE
        )
        self.speaking = False

    def on_speech_start(self) -> None:
        self.speaking = True

    def on_speech_end(self) -> None:
        self.speaking = False

    def is_interrupt(self, text: str) -> bool:
        return bool(self._pattern.search(text))

    def should_admit(self, text: str) -> bool:
        """Whether this transcript should reach the router: always when not
        speaking, only a high-confidence interrupt phrase while speaking."""
        if not self.speaking:
            return True
        return self.is_interrupt(text)


class TTSSpeaker:
    """Sentence-streamed speaker: consumes a text stream, synthesizes and
    plays each sentence as soon as it's complete, and supports a hard
    preempt via `stop()`."""

    def __init__(
        self,
        transport: SynthesisTransport,
        *,
        output: Optional[AudioOutput] = None,
        chunk_bytes: int = CHUNK_BYTES,
        mic_gate: Optional[MicGate] = None,
    ) -> None:
        self._transport = transport
        self._owns_output = output is None
        self._output = output if output is not None else SoundDeviceOutput()
        self._chunk_bytes = chunk_bytes
        self._mic_gate = mic_gate
        self.chunk_ms = chunk_bytes / 2 / SAMPLE_RATE * 1000

        # Bounded so a fast synthesizer can't dump an entire utterance into
        # the queue before playback (and a preempt) get a chance to run --
        # backpressure keeps the generation task genuinely in-flight.
        self._audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=8)
        self._stop_event = asyncio.Event()
        self._gen_task: Optional[asyncio.Task] = None
        self._player_task: Optional[asyncio.Task] = None
        self.is_speaking = False
        self.stopped = False
        self.last_write_time: Optional[float] = None

    async def speak(self, text_stream: AsyncIterator[str], *, span: Optional[TurnSpan] = None) -> None:
        """Consume `text_stream`, splitting on sentence boundaries, and play
        each sentence's audio as soon as it is synthesized -- overlapping
        with later tokens still arriving. Returns once playback of the whole
        stream finishes, or once `stop()` preempts it."""
        # Bind this utterance's queue and stop event locally and pass them to
        # the tasks. A cancelled _generate from a PREVIOUS utterance unwinds
        # after speak() has already reassigned these attributes, so a task that
        # reads self._audio_queue in its teardown would push its sentinel into
        # the NEW utterance's queue and stall it.
        queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue(maxsize=8)
        stop_event = asyncio.Event()
        self._audio_queue = queue
        self._stop_event = stop_event
        self.stopped = False
        self.is_speaking = True
        indicator.set_state(indicator.State.TALKING)
        if self._mic_gate is not None:
            self._mic_gate.on_speech_start()

        self._gen_task = asyncio.create_task(
            self._generate(text_stream, span, queue, stop_event)
        )
        self._player_task = asyncio.create_task(self._play(queue, stop_event))
        try:
            await asyncio.gather(self._gen_task, self._player_task)
        except asyncio.CancelledError:
            self.stopped = True
        finally:
            self.is_speaking = False
            # In `finally` so a preempt or a synthesis error can't leave the
            # indicator stuck reading "talking" with nothing coming out.
            indicator.set_state(indicator.State.IDLE)
            if self._mic_gate is not None:
                self._mic_gate.on_speech_end()
            if self._owns_output:
                self._output.close()

    def stop(self) -> None:
        """Hard preempt: drop queued audio, cancel in-flight synthesis, and
        cancel the upstream text stream. Not a graceful wind-down."""
        self._stop_event.set()
        while True:
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        for task in (self._gen_task, self._player_task):
            if task is not None and not task.done():
                task.cancel()

    async def _generate(
        self,
        text_stream: AsyncIterator[str],
        span: Optional[TurnSpan],
        queue: "asyncio.Queue[Optional[bytes]]",
        stop_event: asyncio.Event,
    ) -> None:
        rechunker = _Rechunker(self._chunk_bytes)
        buffer = ""
        tts_started = False
        try:
            async for token in text_stream:
                if stop_event.is_set():
                    return
                buffer += token
                sentences, buffer = split_ready_sentences(buffer)
                for sentence in sentences:
                    if not tts_started:
                        if span is not None:
                            span.mark("tts_started")
                        tts_started = True
                    await self._synth_sentence(sentence, rechunker, span, queue, stop_event)
                    if stop_event.is_set():
                        return
            remainder = buffer.strip()
            if remainder:
                if not tts_started:
                    if span is not None:
                        span.mark("tts_started")
                    tts_started = True
                await self._synth_sentence(remainder, rechunker, span, queue, stop_event)
            final = rechunker.flush()
            if final is not None and not stop_event.is_set():
                await queue.put(final)
        finally:
            if not stop_event.is_set():
                await queue.put(None)

    async def _synth_sentence(
        self,
        sentence: str,
        rechunker: _Rechunker,
        span: Optional[TurnSpan],
        queue: "asyncio.Queue[Optional[bytes]]",
        stop_event: asyncio.Event,
    ) -> None:
        async for raw in self._transport.synthesize(sentence):
            if stop_event.is_set():
                return
            for chunk in rechunker.feed(raw):
                if stop_event.is_set():
                    return
                if span is not None and "first_content_audio" not in span.stages:
                    span.mark("first_content_audio")
                await queue.put(chunk)

    async def _play(
        self,
        queue: "asyncio.Queue[Optional[bytes]]",
        stop_event: asyncio.Event,
    ) -> None:
        while True:
            chunk = await queue.get()
            if chunk is None or stop_event.is_set():
                return
            await asyncio.to_thread(self._output.write, chunk)
            self.last_write_time = time.perf_counter()


def _require_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        print(
            "error: DEEPGRAM_API_KEY is not set. Export DEEPGRAM_API_KEY=<your key> and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key


def main() -> int:
    """Minimal smoke test: speak a fixed line via real Aura synthesis."""
    api_key = _require_api_key()

    async def _run() -> None:
        async def _one_sentence() -> AsyncIterator[str]:
            yield "This is a Friday TTS smoke test."

        async with DeepgramSpeakTransport(api_key) as transport:
            speaker = TTSSpeaker(transport)
            await speaker.speak(_one_sentence())

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
