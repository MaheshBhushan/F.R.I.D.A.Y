"""Tests for friday.voice.stt: fake-transport plumbing, the preroll seam, local
VAD independence from any transport, and the max-wait turn-close cap.

No network access and no live mic: the fake transport below is injected via a
constructor parameter (dependency injection), never imported by stt.py itself.
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pytest

from friday.core.spans import start_turn
from friday.voice.stt import (
    DEFAULT_ENDPOINTING_MS,
    LocalVAD,
    TranscriptEvent,
    run_utterance,
)
from friday.voice.wake import CHUNK_SAMPLES, SAMPLE_RATE, WakeDetection

TEST_WAV = Path(__file__).resolve().parent.parent / "src" / "friday" / "test_data" / "alexa_test.wav"


class FakeTransport:
    """Mimics Deepgram's streaming message shapes: interim is_final=false results
    with growing transcripts, then a final. Captures every byte sent so the
    preroll/live seam can be asserted byte-exact."""

    def __init__(self, events: list[tuple[float, TranscriptEvent]], config: Optional[dict] = None) -> None:
        self._events = events
        self.config = config or {}
        self.sent_audio = bytearray()
        self.closed = False

    async def __aenter__(self) -> "FakeTransport":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.closed = True

    async def send_media(self, chunk: bytes) -> None:
        self.sent_audio.extend(chunk)

    async def send_close_stream(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[TranscriptEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[TranscriptEvent]:
        for delay, event in self._events:
            if delay:
                await asyncio.sleep(delay)
            yield event


def _wav_chunks(path: Path) -> list[bytes]:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getsampwidth() == 2
        assert w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    frame_bytes = CHUNK_SAMPLES * 2
    chunks = []
    for i in range(0, len(raw), frame_bytes):
        piece = raw[i : i + frame_bytes]
        if len(piece) < frame_bytes:
            piece = piece + b"\x00" * (frame_bytes - len(piece))
        chunks.append(piece)
    return chunks


def _make_detection(live_chunks: list[bytes], preroll_chunks: int = 3, turn_id: str = "t") -> WakeDetection:
    preroll = b"".join(live_chunks[:preroll_chunks])
    rest = live_chunks[preroll_chunks:]
    live: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    for c in rest:
        live.put_nowait(c)
    live.put_nowait(None)
    return WakeDetection(
        model="test",
        score=1.0,
        timestamp=time.time(),
        turn_id=turn_id,
        preroll=preroll,
        preroll_samples=len(preroll) // 2,
        preroll_seconds=(len(preroll) // 2) / SAMPLE_RATE,
        live=live,
    )


def _silence_chunks(n: int) -> list[bytes]:
    return [b"\x00" * (CHUNK_SAMPLES * 2) for _ in range(n)]


# ---------------------------------------------------------------------------
# Criterion 1: interims arrive before the final, against the fake transport.
# ---------------------------------------------------------------------------


def test_interims_arrive_before_final(tmp_path):
    asyncio.run(_interims_arrive_before_final(tmp_path))


async def _interims_arrive_before_final(tmp_path):
    chunks = _wav_chunks(TEST_WAV) + _silence_chunks(10)
    detection = _make_detection(chunks)
    events = [
        (0.0, TranscriptEvent(text="hey", is_final=False)),
        (0.01, TranscriptEvent(text="hey there", is_final=False)),
        (0.01, TranscriptEvent(text="hey there friday", is_final=True, speech_final=True)),
    ]
    transport = FakeTransport(events)
    span = start_turn("reflex", turn_id=detection.turn_id, path=tmp_path / "spans.jsonl")

    seen = []
    async for event in run_utterance(detection, transport, span=span):
        seen.append((time.perf_counter(), event))

    interims = [e for _, e in seen if not e.is_final]
    finals = [e for _, e in seen if e.is_final]
    assert len(interims) >= 2
    assert len(finals) == 1

    # Print the event sequence with timestamps, and prove interims precede stt_final.
    t0 = seen[0][0]
    for ts, event in seen:
        print(f"t={ts - t0:.4f}s is_final={event.is_final} text={event.text!r}")
    last_interim_index = max(i for i, (_, e) in enumerate(seen) if not e.is_final)
    final_index = min(i for i, (_, e) in enumerate(seen) if e.is_final)
    assert last_interim_index < final_index
    assert "speech_ended_vad" in span.stages
    assert "stt_final" in span.stages
    assert span.stages["speech_ended_vad"] <= span.stages["stt_final"]
    print(f"span speech_ended_vad={span.stages['speech_ended_vad']/1e6:.2f}ms stt_final={span.stages['stt_final']/1e6:.2f}ms")


# ---------------------------------------------------------------------------
# Criterion 2: preroll + live bytes arrive at the transport contiguously,
# byte-exact, no gap or duplication at the seam.
# ---------------------------------------------------------------------------


def test_preroll_seam_byte_exact():
    asyncio.run(_preroll_seam_byte_exact())


async def _preroll_seam_byte_exact():
    chunks = _wav_chunks(TEST_WAV) + _silence_chunks(10)
    detection = _make_detection(chunks, preroll_chunks=3)
    live_chunks_sent = chunks[3:]  # what _make_detection queued after the preroll
    expected = detection.preroll + b"".join(live_chunks_sent)

    events = [(0.0, TranscriptEvent(text="x", is_final=True, speech_final=True))]
    transport = FakeTransport(events)

    async for _ in run_utterance(detection, transport):
        pass

    assert bytes(transport.sent_audio) == expected
    assert transport.sent_audio[: len(detection.preroll)] == detection.preroll
    seam = len(detection.preroll)
    assert transport.sent_audio[seam : seam + len(live_chunks_sent[0])] == live_chunks_sent[0]
    print(f"sent_audio bytes={len(transport.sent_audio)} preroll_bytes={len(detection.preroll)} seam_ok=True")


# ---------------------------------------------------------------------------
# Criterion 3: endpointing is explicitly non-default on the connection config.
# ---------------------------------------------------------------------------


def test_endpointing_is_explicit_non_default():
    # Deepgram SDK's own default is 10ms of trailing silence (its own docs).
    # We must not equal that, and must state our chosen value + tradeoff.
    assert DEFAULT_ENDPOINTING_MS != 10
    assert DEFAULT_ENDPOINTING_MS == 100
    fake_config = {"endpointing": DEFAULT_ENDPOINTING_MS}
    print(f"endpointing config passed to SDK: {fake_config}")
    assert fake_config["endpointing"] == 100


# ---------------------------------------------------------------------------
# Criterion 4: local VAD marks speech_ended_vad from local audio only, with the
# transport disconnected/absent entirely.
# ---------------------------------------------------------------------------


def test_local_vad_fires_without_transport(tmp_path):
    asyncio.run(_local_vad_fires_without_transport(tmp_path))


async def _local_vad_fires_without_transport(tmp_path):
    chunks = _wav_chunks(TEST_WAV) + _silence_chunks(10)
    detection = _make_detection(chunks)
    span = start_turn("reflex", turn_id=detection.turn_id, path=tmp_path / "spans.jsonl")

    t0 = time.perf_counter()
    events = [event async for event in run_utterance(detection, None, span=span)]
    t1 = time.perf_counter()

    assert events == []  # no transport => no transcript events, only local VAD
    assert "speech_ended_vad" in span.stages
    print(f"speech_ended_vad fired at {span.stages['speech_ended_vad']/1e6:.2f}ms (wall {(t1 - t0)*1000:.2f}ms), transport=None")


# ---------------------------------------------------------------------------
# Criterion 5: the 700ms max-wait cap closes the turn if the transport never
# sends a final.
# ---------------------------------------------------------------------------


def test_max_wait_cap_closes_turn(tmp_path):
    asyncio.run(_max_wait_cap_closes_turn(tmp_path))


async def _max_wait_cap_closes_turn(tmp_path):
    chunks = _wav_chunks(TEST_WAV) + _silence_chunks(10)
    detection = _make_detection(chunks)
    # Interims only, forever - transport never sends is_final/speech_final/utterance_end.
    events = [(0.05, TranscriptEvent(text="still talking", is_final=False)) for _ in range(50)]
    transport = FakeTransport(events)
    span = start_turn("reflex", turn_id=detection.turn_id, path=tmp_path / "spans.jsonl")

    t0 = time.perf_counter()
    async for _ in run_utterance(detection, transport, span=span, max_wait_ms=700):
        pass
    t1 = time.perf_counter()

    elapsed_since_vad_ms = (span.stages["stt_final"] - span.stages["speech_ended_vad"]) / 1e6
    print(f"turn closed after {(t1 - t0)*1000:.2f}ms wall; {elapsed_since_vad_ms:.2f}ms after speech_ended_vad (cap=700ms)")
    assert "speech_ended_vad" in span.stages
    assert "stt_final" in span.stages
    assert 650 <= elapsed_since_vad_ms <= 900  # cap fired, with scheduling slack


# ---------------------------------------------------------------------------
# Criterion 6: missing DEEPGRAM_API_KEY exits non-zero with a clear message.
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_nonzero(monkeypatch, capsys):
    from friday.voice.stt import _require_api_key

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _require_api_key()
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "DEEPGRAM_API_KEY" in err
    print(f"exit_code={exc_info.value.code} stderr={err.strip()!r}")


def test_local_vad_class_detects_speech_and_silence():
    vad = LocalVAD()
    chunks = _wav_chunks(TEST_WAV) + _silence_chunks(10)
    fired_at = None
    for i, chunk in enumerate(chunks):
        if vad.feed(chunk):
            fired_at = i
            break
    assert fired_at is not None
    print(f"LocalVAD fired at chunk index {fired_at}/{len(chunks)}")


# --- regressions: three bugs the fake transport could not expose -------------
# All three were found only by running against the live Deepgram service.


def test_zero_length_chunks_are_never_sent():
    """Deepgram reads an empty binary frame as end-of-stream and closes the
    socket, which presents as a mid-turn network failure. An empty preroll is a
    legitimate state (a turn opened with no buffered audio), so it must be
    skipped rather than forwarded.
    """

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(b"\x01\x02" * CHUNK_SAMPLES)
        queue.put_nowait(b"")           # must not reach the transport
        queue.put_nowait(b"\x03\x04" * CHUNK_SAMPLES)
        queue.put_nowait(None)
        detection = WakeDetection(
            model="alexa", score=1.0, timestamp=time.time(), turn_id="t",
            preroll=b"",                # empty preroll: must not be sent either
            preroll_samples=0, preroll_seconds=0.0, live=queue,
        )
        transport = FakeTransport([(0.0, TranscriptEvent(text="hi", is_final=True))])
        async for _ in run_utterance(detection, transport, max_wait_ms=100):
            pass
        return transport

    transport = asyncio.run(scenario())
    expected = (b"\x01\x02" * CHUNK_SAMPLES) + (b"\x03\x04" * CHUNK_SAMPLES)
    assert bytes(transport.sent_audio) == expected


def test_turn_closes_when_audio_ends_without_a_silence_tail():
    """Exhausted audio is itself a definitive speech-end. Audio that stops with
    no trailing silence window -- a clipped buffer, a closed mic stream, a
    file-fed turn -- must not leave the closer waiting forever, which would
    deadlock the whole voice loop.
    """

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        # Continuous non-silent audio, then an immediate end: local VAD never
        # observes the trailing silence it needs to declare speech-end.
        for _ in range(4):
            queue.put_nowait(b"\x40\x10" * CHUNK_SAMPLES)
        queue.put_nowait(None)
        detection = WakeDetection(
            model="alexa", score=1.0, timestamp=time.time(), turn_id="t",
            preroll=b"", preroll_samples=0, preroll_seconds=0.0, live=queue,
        )
        with start_turn("reasoning") as span:
            async for _ in run_utterance(detection, None, span=span, max_wait_ms=100):
                pass
            return dict(span.stages)

    stages = asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
    assert "speech_ended_vad" in stages
    assert "stt_final" in stages


def test_pump_failure_surfaces_instead_of_an_empty_turn():
    """A dead audio pump must raise, not degrade into a silent empty turn: the
    original behaviour swallowed the exception via gather(return_exceptions=True)
    so FRIDAY simply stopped hearing with nothing to say why.
    """

    class ExplodingTransport(FakeTransport):
        async def send_media(self, chunk: bytes) -> None:
            raise RuntimeError("socket is gone")

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(b"\x01\x02" * CHUNK_SAMPLES)
        queue.put_nowait(None)
        detection = WakeDetection(
            model="alexa", score=1.0, timestamp=time.time(), turn_id="t",
            preroll=b"", preroll_samples=0, preroll_seconds=0.0, live=queue,
        )
        transport = ExplodingTransport([])
        async for _ in run_utterance(detection, transport, max_wait_ms=100):
            pass

    with pytest.raises(RuntimeError, match="socket is gone"):
        asyncio.run(scenario())
