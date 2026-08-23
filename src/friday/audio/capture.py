"""Sequence-numbered, fixed-frame audio fan-out."""

from __future__ import annotations

import asyncio
import math
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from friday.audio.manager import AudioResourceManager

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1_000 * 2
RING_FRAMES = 2_000 // FRAME_MS
WAKE_FRAMES = 80 // FRAME_MS


@dataclass(frozen=True, slots=True)
class AudioFrame:
    sequence: int
    timestamp_ns: int
    pcm: bytes


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    frames_captured: int
    duration_seconds: float
    rms_db: float
    bytes_captured: int
    partial_bytes: int
    ring_frames: int
    active_subscribers: int
    last_sequence: int


class AudioSubscription:
    """A frozen pre-roll snapshot followed by live frames, each exactly once."""

    def __init__(
        self,
        snapshot: tuple[AudioFrame, ...],
        remove: Callable[["AudioSubscription"], None],
    ) -> None:
        self._snapshot = snapshot
        self._pending = deque(snapshot)
        self._queue: "asyncio.Queue[Optional[AudioFrame]]" = asyncio.Queue()
        self._remove = remove
        self._closed = False

    @property
    def snapshot(self) -> tuple[AudioFrame, ...]:
        return self._snapshot

    def __aiter__(self) -> "AudioSubscription":
        return self

    async def __anext__(self) -> AudioFrame:
        if self._pending:
            return self._pending.popleft()
        if self._closed:
            raise StopAsyncIteration
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def __aenter__(self) -> "AudioSubscription":
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._snapshot = ()
        self._pending.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(None)
        self._remove(self)

    def _publish(self, frame: AudioFrame) -> None:
        if not self._closed:
            self._queue.put_nowait(frame)


class AudioCaptureService:
    """Own the manager stream and fan it out as 20ms mono PCM frames."""

    def __init__(
        self,
        manager: AudioResourceManager,
        *,
        ring_frames: int = RING_FRAMES,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._manager = manager
        self._clock_ns = clock_ns
        self._ring: deque[AudioFrame] = deque(maxlen=ring_frames)
        self._partial = bytearray()
        self._subscribers: set[AudioSubscription] = set()
        self._next_sequence = 0
        self._last_sequence = -1
        self._frames_captured = 0
        self._bytes_captured = 0
        self._sample_count = 0
        self._sample_square_sum = 0
        self._running = False

    @property
    def metrics(self) -> AudioMetrics:
        rms = (math.sqrt(self._sample_square_sum / self._sample_count)
               if self._sample_count else 0.0)
        rms_db = 20 * math.log10(rms / 32_768) if rms else float("-inf")
        return AudioMetrics(
            frames_captured=self._frames_captured,
            duration_seconds=self._frames_captured * FRAME_MS / 1_000,
            rms_db=rms_db,
            bytes_captured=self._bytes_captured,
            partial_bytes=len(self._partial),
            ring_frames=len(self._ring),
            active_subscribers=len(self._subscribers),
            last_sequence=self._last_sequence,
        )

    @property
    def manager(self) -> AudioResourceManager:
        return self._manager

    def snapshot(self) -> tuple[AudioFrame, ...]:
        """Freeze the current approximately two-second ring chronologically."""
        return tuple(self._ring)

    def subscribe_wake(self) -> AudioSubscription:
        return self._subscribe(())

    def subscribe_stt(self) -> AudioSubscription:
        """Attach immediately, before an STT connector performs any awaits."""
        return self._subscribe(self.snapshot())

    def subscribe_live(self) -> AudioSubscription:
        """Attach without pre-roll for a wake-free conversation follow-up."""
        return self._subscribe(())

    async def wake_chunks(self) -> AsyncIterator[bytes]:
        """Yield the 80ms chunks required by the existing wake detector."""
        subscription = self.subscribe_wake()
        frames: list[bytes] = []
        try:
            async for frame in subscription:
                frames.append(frame.pcm)
                if len(frames) == WAKE_FRAMES:
                    yield b"".join(frames)
                    frames.clear()
        finally:
            subscription.close()

    async def run(self, stop: asyncio.Event) -> None:
        if self._running:
            raise RuntimeError("audio capture service is already running")
        self._running = True
        try:
            async for chunk in self._manager.capture(stop):
                self._ingest(chunk)
        finally:
            self._running = False

    def reset(self) -> None:
        """Drop all retained audio; suitable for the manager's on_forget hook."""
        for subscription in tuple(self._subscribers):
            subscription.close()
        self._ring.clear()
        self._partial.clear()
        self._last_sequence = -1
        self._frames_captured = 0
        self._bytes_captured = 0
        self._sample_count = 0
        self._sample_square_sum = 0

    def _subscribe(self, snapshot: tuple[AudioFrame, ...]) -> AudioSubscription:
        subscription = AudioSubscription(snapshot, self._subscribers.discard)
        self._subscribers.add(subscription)
        return subscription

    def _ingest(self, chunk: bytes) -> None:
        self._partial.extend(chunk)
        while len(self._partial) >= FRAME_BYTES:
            pcm = bytes(self._partial[:FRAME_BYTES])
            del self._partial[:FRAME_BYTES]
            frame = AudioFrame(self._next_sequence, self._clock_ns(), pcm)
            self._next_sequence += 1
            self._last_sequence = frame.sequence
            self._frames_captured += 1
            self._bytes_captured += len(pcm)
            samples = struct.iter_unpack("<h", pcm)
            self._sample_square_sum += sum(sample * sample for sample, in samples)
            self._sample_count += len(pcm) // 2
            self._ring.append(frame)
            for subscription in tuple(self._subscribers):
                subscription._publish(frame)
