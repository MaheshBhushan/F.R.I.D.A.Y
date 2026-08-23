from __future__ import annotations

import asyncio
import math

from friday.audio.capture import (
    FRAME_BYTES,
    RING_FRAMES,
    AudioCaptureService,
)


class Source:
    def __init__(self, chunks=()) -> None:
        self.chunks = chunks

    async def capture(self, stop):
        for chunk in self.chunks:
            yield chunk


def _service(chunks=(), **kwargs) -> AudioCaptureService:
    return AudioCaptureService(Source(chunks), **kwargs)


def test_manager_chunks_are_normalized_to_sequenced_20ms_frames():
    raw = bytes(range(256)) * 8
    service = _service((raw[:17], raw[17:701], raw[701:]))

    asyncio.run(service.run(asyncio.Event()))

    frames = service.snapshot()
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert all(len(frame.pcm) == FRAME_BYTES for frame in frames)
    assert b"".join(frame.pcm for frame in frames) == raw[:3 * FRAME_BYTES]
    assert service.metrics.partial_bytes == len(raw) - 3 * FRAME_BYTES


def test_ring_snapshot_is_frozen_chronological_and_two_seconds_long():
    service = _service()
    service._ingest(b"".join(i.to_bytes(2, "little") * (FRAME_BYTES // 2)
                              for i in range(RING_FRAMES + 5)))

    snapshot = service.snapshot()
    service._ingest(b"\xff\xff" * (FRAME_BYTES // 2))

    assert isinstance(snapshot, tuple)
    assert [frame.sequence for frame in snapshot] == list(range(5, 105))
    assert [frame.sequence for frame in service.snapshot()] == list(range(6, 106))


def test_frames_use_the_injected_monotonic_nanosecond_clock():
    timestamps = iter((1_000, 2_000, 3_000))
    service = _service(clock_ns=lambda: next(timestamps))

    service._ingest(b"t" * FRAME_BYTES * 3)

    assert [frame.timestamp_ns for frame in service.snapshot()] == [1_000, 2_000, 3_000]


def test_metrics_report_captured_duration_and_aggregate_rms_db():
    half_scale = (16_384).to_bytes(2, "little", signed=True)
    service = _service()

    service._ingest(half_scale * (FRAME_BYTES // 2) * 2)

    assert service.metrics.frames_captured == 2
    assert service.metrics.duration_seconds == 0.04
    assert math.isclose(service.metrics.rms_db, -6.020599913, abs_tol=1e-9)


def test_stt_subscription_retains_frames_while_connector_waits():
    async def _run():
        service = _service()
        service._ingest(b"a" * FRAME_BYTES * 2)
        subscription = service.subscribe_stt()

        await asyncio.sleep(0)  # simulated network connection await
        service._ingest(b"b" * FRAME_BYTES * 2)

        seen = [await anext(subscription) for _ in range(4)]
        assert [frame.sequence for frame in seen] == [0, 1, 2, 3]
        assert b"".join(frame.pcm for frame in seen) == (
            b"a" * FRAME_BYTES * 2 + b"b" * FRAME_BYTES * 2)
        subscription.close()

    asyncio.run(_run())


def test_preroll_pending_and_live_frames_are_consumed_once_without_duplicates():
    async def _run():
        service = _service()
        service._ingest(b"p" * FRAME_BYTES * 2)
        subscription = service.subscribe_stt()
        service._ingest(b"q" * FRAME_BYTES)

        first = [await anext(subscription) for _ in range(3)]
        assert [frame.sequence for frame in subscription.snapshot] == [0, 1]
        service._ingest(b"r" * FRAME_BYTES)
        second = await anext(subscription)

        assert [frame.sequence for frame in [*first, second]] == [0, 1, 2, 3]
        subscription.close()

    asyncio.run(_run())


def test_wake_consumer_receives_80ms_chunks_from_20ms_frames():
    async def _run():
        service = _service()
        chunks = service.wake_chunks()
        pending = asyncio.create_task(anext(chunks))
        await asyncio.sleep(0)
        service._ingest(b"w" * FRAME_BYTES * 4)

        assert await pending == b"w" * FRAME_BYTES * 4
        await chunks.aclose()

    asyncio.run(_run())


def test_reset_clears_ring_subscribers_partial_audio_and_metrics():
    async def _run():
        service = _service()
        service._ingest(b"x" * (FRAME_BYTES + 7))
        subscription = service.subscribe_stt()
        assert service.metrics.active_subscribers == 1

        service.reset()

        assert service.snapshot() == ()
        assert service.metrics.frames_captured == 0
        assert service.metrics.duration_seconds == 0
        assert service.metrics.rms_db == float("-inf")
        assert service.metrics.bytes_captured == 0
        assert service.metrics.partial_bytes == 0
        assert service.metrics.active_subscribers == 0
        assert service.metrics.last_sequence == -1
        assert [frame async for frame in subscription] == []

        service._ingest(b"y" * FRAME_BYTES)
        assert service.snapshot()[0].sequence == 1

    asyncio.run(_run())
