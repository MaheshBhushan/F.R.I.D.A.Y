"""Tests for friday.voice.tts: sentence streaming overlap, hard-preempt
latency and cleanliness, mic gating, chunk sizing, and the missing-key path.

No network access and no audio hardware: FakeSynthesisTransport is injected
the same way stt.py's fake Transport is, and playback goes to an in-memory
fake AudioOutput.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import statistics
import time
from typing import AsyncIterator, List, Tuple

import pytest

from friday.core.spans import start_turn
from friday.voice.tts import (
    CHUNK_BYTES,
    CHUNK_MS,
    SAMPLE_RATE,
    FakeSynthesisTransport,
    MicGate,
    TTSSpeaker,
    _require_api_key,
    split_ready_sentences,
)


class RecordingOutput:
    """In-memory fake AudioOutput: records every write with a wall-clock
    timestamp, no real device involved."""

    def __init__(self, write_sleep: float = 0.0) -> None:
        self.write_sleep = write_sleep
        self.writes: List[Tuple[float, int]] = []
        self.closed = False

    def write(self, chunk: bytes) -> None:
        if self.write_sleep:
            time.sleep(self.write_sleep)
        self.writes.append((time.perf_counter(), len(chunk)))

    def close(self) -> None:
        self.closed = True


async def _stream_from(sentences: List[str], delay: float, events: list) -> AsyncIterator[str]:
    """Fake LLM token stream: yields one sentence's words at a time with a
    delay between sentences, and logs when it (the stream) is exhausted."""
    for i, sentence in enumerate(sentences):
        if i > 0:
            await asyncio.sleep(delay)
        for word in sentence.split(" "):
            yield word + " "
    events.append(("stream_exhausted", time.perf_counter()))


# ---------------------------------------------------------------------------
# 1. Sentence streaming: first audio precedes generation completion
# ---------------------------------------------------------------------------


def test_sentence_splitter_extracts_ready_sentences_only():
    sentences, remainder = split_ready_sentences("Hello world. How are you")
    assert sentences == ["Hello world."]
    assert remainder == "How are you"


def test_first_audio_precedes_stream_exhaustion():
    async def scenario():
        events: list = []
        sentences = [f"Sentence number {i}." for i in range(5)]
        transport = FakeSynthesisTransport(sentence_delay=0.0, bytes_per_sentence=SAMPLE_RATE)
        output = RecordingOutput()
        speaker = TTSSpeaker(transport, output=output, chunk_bytes=CHUNK_BYTES)
        span = start_turn("reasoning", turn_id="t6-overlap-test")

        text_stream = _stream_from(sentences, delay=0.05, events=events)
        await speaker.speak(text_stream, span=span)
        return events, span, transport

    events, span, transport = asyncio.run(scenario())

    assert "tts_started" in span.stages
    assert "first_content_audio" in span.stages
    assert len(transport.synthesized) == 5

    stream_exhausted_ns = next(t for name, t in events if name == "stream_exhausted")
    # first_content_audio is a monotonic offset from the span's own start;
    # stream_exhausted_ns is wall-clock from time.perf_counter(). Convert the
    # span offset to the same clock isn't directly possible, so instead prove
    # overlap the way the criterion asks: audio started well before the last
    # of 5 sentences (spaced 50ms apart, ~200ms total) was even requested.
    first_audio_offset_s = span.stages["first_content_audio"] / 1e9
    tts_started_offset_s = span.stages["tts_started"] / 1e9
    print("\n--- event timeline (criterion 1) ---")
    print(f"tts_started        @ {tts_started_offset_s*1000:.1f}ms (offset from turn start)")
    print(f"first_content_audio @ {first_audio_offset_s*1000:.1f}ms (offset from turn start)")
    print(f"5 sentences requested with 50ms spacing => ~200ms total generation time")
    print(f"span record: {span.to_record()}")

    # The whole point: first_content_audio fires immediately after the FIRST
    # sentence synthesizes, not after all 5 have been requested/generated.
    assert first_audio_offset_s < 0.15, "first audio should precede later sentences, not wait for all of them"
    assert len(transport.synthesized) == 5


# ---------------------------------------------------------------------------
# 2. Hard preempt within 80ms
# ---------------------------------------------------------------------------


def _run_one_preempt(chunk_sleep_s: float, wait_before_stop_s: float) -> tuple:
    async def scenario():
        transport = FakeSynthesisTransport(bytes_per_sentence=CHUNK_BYTES * 200)
        output = RecordingOutput(write_sleep=chunk_sleep_s)
        speaker = TTSSpeaker(transport, output=output, chunk_bytes=CHUNK_BYTES)

        async def stream() -> AsyncIterator[str]:
            yield "Hello there, this is a long sentence that keeps going. "

        task = asyncio.create_task(speaker.speak(stream()))
        # Vary the moment we preempt relative to the write cadence so some
        # calls land mid-write (the physically-limited worst case: a write
        # already in flight on its own thread cannot be aborted, only
        # awaited) and some land in the gap between writes.
        await asyncio.sleep(wait_before_stop_s)

        t_stop = time.perf_counter()
        speaker.stop()
        await task
        # Cancelling the player task only stops it from awaiting a new
        # write; a write already handed to a background thread (as a real
        # device write would be) keeps running there and appends when it's
        # done. Give it a moment to land before reading the last byte time,
        # or this would race the very thread we're trying to measure.
        await asyncio.sleep(chunk_sleep_s * 1.5)

        last_write = output.writes[-1][0] if output.writes else t_stop
        latency_ms = max(0.0, (last_write - t_stop) * 1000)
        return latency_ms, speaker, output

    return asyncio.run(scenario())


def test_hard_preempt_latency_and_cleanup():
    chunk_sleep_s = CHUNK_MS / 1000.0  # simulate real-time playback per chunk
    rng = random.Random(0)
    latencies = []
    for _ in range(25):
        wait_before_stop_s = chunk_sleep_s * 2 + rng.uniform(0, chunk_sleep_s)
        latency_ms, speaker, output = _run_one_preempt(chunk_sleep_s, wait_before_stop_s)
        latencies.append(latency_ms)
        # The task doing "async for token in text_stream" must actually be
        # cancelled, not merely ignored.
        assert speaker._gen_task.cancelled(), "upstream generation task was not cancelled"
        assert speaker._audio_queue.empty(), "audio queue was not drained on preempt"

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(0.90 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]
    print("\n--- hard preempt latency over 25 runs (criterion 2) ---")
    print(f"p50={p50:.1f}ms p90={p90:.1f}ms p99={p99:.1f}ms  all={[round(x,1) for x in latencies]}")

    assert p50 < 80
    assert p90 < 80
    assert p99 < 80


# ---------------------------------------------------------------------------
# 3. Mic gating during playback
# ---------------------------------------------------------------------------


def test_mic_gate_blocks_ordinary_speech_admits_interrupt_while_speaking():
    gate = MicGate()
    assert gate.should_admit("what's the weather today") is True  # not speaking yet

    gate.on_speech_start()
    assert gate.speaking is True
    assert gate.should_admit("what's the weather today") is False
    assert gate.should_admit("please stop") is True
    assert gate.should_admit("wait a second") is True
    assert gate.is_interrupt("please stop") is True

    gate.on_speech_end()
    assert gate.speaking is False
    assert gate.should_admit("what's the weather today") is True


def test_mic_gate_wired_to_speaker_stop_on_interrupt():
    async def scenario():
        transport = FakeSynthesisTransport(bytes_per_sentence=CHUNK_BYTES * 200)
        output = RecordingOutput(write_sleep=CHUNK_MS / 1000.0)
        gate = MicGate()
        speaker = TTSSpeaker(transport, output=output, chunk_bytes=CHUNK_BYTES, mic_gate=gate)

        async def stream() -> AsyncIterator[str]:
            yield "A very long sentence that keeps on going and going. "

        task = asyncio.create_task(speaker.speak(stream()))
        await asyncio.sleep(0.05)
        assert gate.speaking is True

        ordinary_admitted = gate.should_admit("turn on the lights")
        interrupt_admitted = gate.should_admit("stop")
        if interrupt_admitted:
            speaker.stop()
        await task
        return ordinary_admitted, interrupt_admitted, speaker

    ordinary_admitted, interrupt_admitted, speaker = asyncio.run(scenario())
    assert ordinary_admitted is False
    assert interrupt_admitted is True
    assert speaker.stopped is True


# ---------------------------------------------------------------------------
# 4. Chunk size
# ---------------------------------------------------------------------------


def test_chunk_size_within_40_to_80ms():
    duration_ms = CHUNK_BYTES / 2 / SAMPLE_RATE * 1000  # 16-bit samples
    print(f"\n--- chunk size (criterion 4) ---\nCHUNK_BYTES={CHUNK_BYTES} -> {duration_ms:.1f}ms @ {SAMPLE_RATE}Hz")
    assert 40 <= duration_ms <= 80


def test_speaker_actual_chunks_match_declared_size():
    async def scenario():
        transport = FakeSynthesisTransport(bytes_per_sentence=CHUNK_BYTES * 3 + 100)
        output = RecordingOutput()
        speaker = TTSSpeaker(transport, output=output, chunk_bytes=CHUNK_BYTES)

        async def stream() -> AsyncIterator[str]:
            yield "One sentence."

        await speaker.speak(stream())
        return output

    output = asyncio.run(scenario())
    assert output.writes, "expected at least one chunk written"
    for _, size in output.writes:
        assert size == CHUNK_BYTES


def test_speaker_closes_the_output_it_creates(monkeypatch):
    output = RecordingOutput()
    monkeypatch.setattr("friday.voice.tts.SoundDeviceOutput", lambda: output)

    async def scenario():
        async def stream() -> AsyncIterator[str]:
            yield "One sentence."

        speaker = TTSSpeaker(FakeSynthesisTransport())
        await speaker.speak(stream())

    asyncio.run(scenario())
    assert output.closed is True


# ---------------------------------------------------------------------------
# 5. Missing DEEPGRAM_API_KEY
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_nonzero_naming_the_variable(monkeypatch, capsys):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _require_api_key()
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    print(f"\n--- missing key output (criterion 5) ---\n{captured.err}")
    assert "DEEPGRAM_API_KEY" in captured.err


class _CountingOutput:
    def __init__(self):
        self.count = 0

    def write(self, chunk):
        self.count += 1

    def close(self):
        pass


# --- regression: reuse immediately after stop() ------------------------------

def test_speaker_is_reusable_immediately_after_stop():
    """A preempted utterance's cancelled tasks unwind after speak() has already
    rebound the queue. If their teardown pushes the sentinel into the NEW
    utterance's queue, the next utterance produces no audio and hangs. This is
    exactly the sequence a proactive interrupt performs.
    """

    async def stream(n, gap=0.005):
        for i in range(n):
            yield f"Sentence {i} here. "
            await asyncio.sleep(gap)

    async def scenario():
        out = _CountingOutput()
        speaker = TTSSpeaker(FakeSynthesisTransport(), output=out)

        first = asyncio.create_task(speaker.speak(stream(200)))
        await asyncio.sleep(0.08)
        speaker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await first

        before = out.count
        # No yield/sleep between stop() and the next speak() -- that is the point.
        await asyncio.wait_for(speaker.speak(stream(3, gap=0.001)), timeout=5.0)
        return before, out.count

    before, after = asyncio.run(scenario())
    assert after > before, "second utterance produced no audio after preempt"
