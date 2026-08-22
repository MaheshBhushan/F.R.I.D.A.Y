"""Local wake-word detection with a pre-roll ring buffer.

Runs a continuous mic capture loop, feeds 80ms PCM frames through
openWakeWord, and on detection hands the downstream speech pipeline a
stream that starts with the buffered pre-roll audio (so cloud STT socket
setup latency never clips the first word of the command) followed by
live audio with no gap and no duplicated samples.

Capture backend: sounddevice (installs cleanly via pip/uv on this
machine's PortAudio; pyaudio needs extra system dev headers to build).
Wake-word engine: openwakeword, pretrained "alexa" model, run through
the onnx inference framework (the tflite_runtime wheel in this venv is
built against numpy 1.x and segfaults/raises under numpy 2.x; onnx has
no such conflict and both runtimes ship in the package).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from openwakeword.model import Model

from friday.core import events
from friday.core.spans import TurnSpan, start_turn

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms @ 16kHz mono int16 - openWakeWord's native frame size
PREROLL_SECONDS = 1.5
PREROLL_CHUNKS = -(-int(PREROLL_SECONDS * SAMPLE_RATE) // CHUNK_SAMPLES)  # ceil
DEFAULT_THRESHOLD = 0.5
# Chunks to buffer without running inference after the detector starts.
# A freshly opened capture stream drops its second frame (measured: chunk 1
# came back at rms 4.7 between neighbours at 302 and 362), and openWakeWord's
# feature window is about a second wide, so that discontinuity reliably scores
# as a wake word 0.72-1.12s later -- a phantom turn on every startup, every
# time. 15 chunks (1.2s) clears the window before the first prediction.
WARMUP_CHUNKS = 15

# Only two pretrained models ship with openwakeword: "alexa" and "hey_jarvis".
# There is no "friday" model, so saying her name does nothing -- the single most
# confusing property of this system, and worth an env var rather than a code
# edit to change. FRIDAY_WAKE_MODEL takes a comma-separated list; every name
# must be one openwakeword can resolve, or Model() raises at construction.
_MODEL_ENV = "FRIDAY_WAKE_MODEL"


# Locally trained models live here, inside the package, so they ship and get
# version-controlled with the code that depends on them. openwakeword resolves
# a bare name only against its own bundled resources, so a custom name has to
# be turned into a path before it ever reaches Model().
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def default_models() -> tuple[str, ...]:
    raw = os.environ.get(_MODEL_ENV, "").strip()
    if not raw:
        return ("alexa",)
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def resolve_models(names: "tuple[str, ...]") -> tuple[str, ...]:
    """Turn wake-model names into what openwakeword's Model() accepts.

    A name matching a file in MODELS_DIR becomes that path; anything else is
    passed through untouched so openwakeword's own "alexa"/"hey_jarvis" keep
    working, and an outright typo still fails loudly inside Model() rather than
    being silently swallowed here.

    An explicit path is honoured as-is, which is what makes it possible to
    A/B a freshly trained model without installing it first.
    """
    resolved = []
    for name in names:
        if os.sep in name or name.endswith(".onnx") or name.endswith(".tflite"):
            resolved.append(name)
            continue
        local = MODELS_DIR / f"{name}.onnx"
        resolved.append(str(local) if local.is_file() else name)
    return tuple(resolved)


def available_models() -> tuple[str, ...]:
    """Locally trained model names, for `friday hear` and error messages."""
    if not MODELS_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in MODELS_DIR.glob("*.onnx")))


DEFAULT_MODELS = ("alexa",)
# Scores below this are silence and noise; logging them at debug would emit
# 12.5 lines a second and bury the near-misses that actually diagnose a
# "wake word not working" report.
SCORE_LOG_FLOOR = 0.10
# Input-level landmarks for the probe's verdict, measured on this machine's
# built-in mic: a silent/dead stream sits near 0, an idle quiet room around
# 100, and actual speech in the low thousands.
DEAD_STREAM_LEVEL = 25.0
SPEECH_LEVEL = 600.0

# openWakeWord keeps 10 seconds of raw audio in a Python deque
# (deque(maxlen=sr*10) = 160,000 int objects) and, once per 80ms frame, does
#     list(self.raw_data_buffer)[-n_samples-480:]
# to get the newest ~1,760 samples. That materialises all 160,000 as Python
# ints and discards 99% of them. Profiled on this machine it was 25% of the
# daemon's entire idle CPU -- more than the melspectrogram model itself.
#
# Nothing in openwakeword reads further back than n_samples+480, so a one
# second buffer keeps a ~9x safety margin over the 1,760 actually needed while
# cutting the per-frame copy by 10x. Measured: 1.51ms -> 0.03ms per frame.
RAW_BUFFER_SECONDS = 1.0
TEST_WAV = Path(__file__).resolve().parent.parent / "test_data" / "alexa_test.wav"


@dataclass
class WakeDetection:
    """Handoff produced the moment a wake word fires.

    `preroll` is the buffered audio (pre-roll, in chunk order) up to and
    including the triggering frame. `live` is an asyncio.Queue that the
    detector keeps pushing subsequent frames into until `end()` is
    called; a consumer should drain `preroll` first, then read `live`
    until it yields None, giving a gapless, non-duplicated stream.
    """

    model: str
    score: float
    timestamp: float
    turn_id: str
    preroll: bytes
    preroll_samples: int
    preroll_seconds: float
    live: "asyncio.Queue[Optional[bytes]]" = field(repr=False)


class WakeWordDetector:
    """Feeds fixed-size PCM frames through openWakeWord over a pre-roll ring buffer."""

    def __init__(
        self,
        model_names: Optional[tuple[str, ...]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        preroll_chunks: int = PREROLL_CHUNKS,
        inference_framework: str = "onnx",
        warmup_chunks: int = WARMUP_CHUNKS,
    ) -> None:
        model_names = tuple(model_names or default_models())
        self.threshold = threshold
        self.warmup_chunks = warmup_chunks
        self.model_names = model_names
        # Warmup starts already satisfied. It is a property of a live capture
        # stream, not of the detector, so file-fed callers (tests, run_bench,
        # --file) stay exact and a wake word in the first 1.2s still fires.
        # `begin_stream()` is what arms it.
        self._fed = warmup_chunks
        self._model = Model(wakeword_models=list(resolve_models(model_names)),
                            inference_framework=inference_framework)
        self._shrink_raw_buffer()
        events.emit("wake-init", models=",".join(self.model_names),
                    threshold=threshold, warmup=warmup_chunks)
        self._ring: deque[bytes] = deque(maxlen=preroll_chunks)
        self._active_live: Optional["asyncio.Queue[Optional[bytes]]"] = None

    def _shrink_raw_buffer(self) -> None:
        """Right-size openWakeWord's raw-audio deque (see RAW_BUFFER_SECONDS).

        Best-effort and silent on failure: this is a performance fix reaching
        into another library's internals, and a future version that renames the
        attribute must cost us CPU, never correctness.
        """
        try:
            pre = self._model.preprocessor
            keep = int(SAMPLE_RATE * RAW_BUFFER_SECONDS)
            if pre.raw_data_buffer.maxlen and pre.raw_data_buffer.maxlen > keep:
                pre.raw_data_buffer = deque(pre.raw_data_buffer, maxlen=keep)
        except Exception:  # noqa: BLE001
            pass

    @property
    def ring_samples(self) -> int:
        return sum(len(c) // 2 for c in self._ring)

    @property
    def ring_seconds(self) -> float:
        return self.ring_samples / SAMPLE_RATE

    def begin_stream(self) -> None:
        """Arm warmup for a newly opened capture stream. Call once per stream."""
        self._fed = 0
        self._ring.clear()

    def feed_chunk(self, chunk: bytes, span: Optional[TurnSpan] = None) -> Optional[WakeDetection]:
        """Feed one CHUNK_SAMPLES-sized frame of raw int16 PCM.

        If a live handoff is in progress (post-detection), the frame is
        forwarded straight to the live queue instead of the wake model,
        so the pre-roll/live seam never repeats or drops a chunk. Returns
        a WakeDetection the instant the wake word fires, else None.
        """
        if self._active_live is not None:
            self._active_live.put_nowait(chunk)
            return None

        self._ring.append(chunk)
        frame = np.frombuffer(chunk, dtype=np.int16)
        # Feed the model during warmup so its buffer is primed, but ignore the
        # verdict: the startup discontinuity is still inside the window.
        self._fed += 1
        scores = self._model.predict(frame)
        if self._fed <= self.warmup_chunks:
            return None
        if scores:
            top, peak = max(scores.items(), key=lambda kv: kv[1])
            if peak >= SCORE_LOG_FLOOR:
                # A near-miss is the whole diagnosis for "it does not hear me":
                # steady 0.3s mean the voice is landing but under threshold,
                # while a flat 0.0 means the word being spoken is not the word
                # the model was trained on.
                events.debug("wake-score", model=top, score=float(peak))
        for name, score in scores.items():
            if score >= self.threshold:
                if span is not None:
                    span.mark("speech_started")
                preroll = b"".join(self._ring)
                events.emit("wake", model=name, score=float(score),
                            preroll=f"{(len(preroll) // 2) / SAMPLE_RATE:.2f}s")
                live: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
                self._active_live = live
                return WakeDetection(
                    model=name,
                    score=float(score),
                    timestamp=time.time(),
                    turn_id=span.turn_id if span is not None else "",
                    preroll=preroll,
                    preroll_samples=len(preroll) // 2,
                    preroll_seconds=(len(preroll) // 2) / SAMPLE_RATE,
                    live=live,
                )
        return None

    def end_handoff(self, live: "Optional[asyncio.Queue[Optional[bytes]]]" = None) -> None:
        """Signal the current live queue is done and resume wake detection.

        Pass `live` to make the release detection-scoped. Turns can overlap
        (a barge-in starts a new handoff while the previous turn is still
        unwinding), and an unscoped release from the OLD turn would close the
        NEW turn's queue, starving it of audio. With `live` given, a stale
        caller is a no-op instead.
        """
        if self._active_live is None:
            return
        if live is not None and self._active_live is not live:
            return
        self._active_live.put_nowait(None)
        self._active_live = None
        self._ring.clear()


async def capture_loop(detector: WakeWordDetector, on_detection, span: Optional[TurnSpan] = None) -> None:
    """Continuously capture mic audio and feed it through `detector`."""
    import sounddevice as sd

    from friday.voice import devices, indicator

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _callback(indata, frames, time_info, status) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    devices.apply()
    indicator.set_state(indicator.State.IDLE)
    detector.begin_stream()
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        dtype="int16",
        channels=1,
        callback=_callback,
        device=devices.input_device(),
    ):
        while True:
            chunk = await queue.get()
            detection = detector.feed_chunk(chunk, span=span)
            if detection is not None:
                # Set state here, not in feed_chunk: run_bench() times that
                # call for the wake-latency gate, and a file write inside it
                # would inflate the measured number.
                indicator.set_state(indicator.State.LISTENING)
                on_detection(detection)


async def probe(seconds: float = 20.0, threshold: float = DEFAULT_THRESHOLD,
                models: "Optional[tuple[str, ...]]" = None) -> int:
    """Live mic probe: print input level and wake score once a second.

    This exists because "the wake word does not work" has three completely
    different causes that look identical from the outside, and this separates
    them in one run:

      level 0, score 0    -> no audio arriving at all (routing/permissions)
      level high, score 0 -> audio is fine, the spoken word is not the model's
      level high, score .3-.5 -> heard, but under threshold (accent/distance)

    It opens its own capture stream alongside the daemon rather than asking the
    daemon for scores. PipeWire multiplexes readers, so this observes the same
    audio without taking the microphone away from her -- a diagnostic that
    silences the thing being diagnosed is useless.
    """
    import numpy as np
    import sounddevice as sd

    from friday.voice import devices

    routing = devices.apply()
    detector = WakeWordDetector(threshold=threshold, model_names=models)
    detector.begin_stream()
    print(f"source: {routing.source or 'default'}"
          f"{'  (degraded: ' + routing.reason + ')' if routing.degraded else ''}")
    print(f"models: {', '.join(detector.model_names)}   threshold: {threshold}")
    print(f"speak the wake word; {seconds:.0f}s\n")
    print(f"{'time':>5}  {'level':>6}  {'score':>6}  best-model")

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[bytes]" = asyncio.Queue()
    sd.default.device = None

    def _callback(indata, frames, time_info, status) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    t0 = time.time()
    peak_score, peak_model, fires, peak_level = 0.0, "-", 0, 0.0
    win_level, win_score, win_model, win_n = 0.0, 0.0, "-", 0
    next_report = 1.0
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                           dtype="int16", channels=1, callback=_callback,
                           device=devices.input_device()):
        while True:
            elapsed = time.time() - t0
            if elapsed >= seconds:
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=seconds - elapsed)
            except asyncio.TimeoutError:
                break
            frame = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
            scores = detector._model.predict(np.frombuffer(chunk, dtype=np.int16))
            detector._fed += 1
            top, score = ("-", 0.0)
            if scores:
                top, score = max(scores.items(), key=lambda kv: kv[1])
                score = float(score)
            warm = detector._fed > detector.warmup_chunks
            win_level = max(win_level, level)
            peak_level = max(peak_level, level)
            win_n += 1
            if warm and score > win_score:
                win_score, win_model = score, top
            if warm and score >= threshold:
                fires += 1
            if warm and score > peak_score:
                peak_score, peak_model = score, top
            if elapsed >= next_report:
                bar = "#" * min(20, int(win_level / 400))
                flag = "  <-- WAKE" if win_score >= threshold else ""
                print(f"{elapsed:5.1f}  {win_level:6.0f}  {win_score:6.3f}  "
                      f"{win_model:<12} {bar}{flag}")
                win_level, win_score, win_model, win_n = 0.0, 0.0, "-", 0
                next_report = elapsed + 1.0

    print(f"\npeak score {peak_score:.3f} ({peak_model}), fires {fires}")
    if fires:
        print("verdict: wake detection works.")
    elif peak_score >= threshold * 0.5:
        print(f"verdict: heard but under threshold ({peak_score:.3f} < {threshold}). "
              f"Speak closer, or lower it.")
    elif peak_level < DEAD_STREAM_LEVEL:
        # A dead stream scores 0.0 exactly like a wrong word does, so it is
        # checked first -- blaming the phrase here sends the user off fixing
        # the wrong thing. Measured on this machine: a genuinely silent stream
        # sits near 0, a quiet room around 100, speech in the thousands.
        print("verdict: no audio reached the detector at all. The wake word is "
              "not the problem -- check routing with `friday doctor`.")
    elif peak_level < SPEECH_LEVEL:
        print(f"verdict: audio is arriving but never rose above room level "
              f"(peak {peak_level:.0f}). Nothing was said loudly enough to "
              f"judge the wake word -- try again, closer to the mic.")
    else:
        print("verdict: no wake-word signal. The input level moved, so the audio "
              "is fine and the word you said is not the word the model knows.")
        print(f"  loaded: {', '.join(detector.model_names)}")
        local = available_models()
        if local:
            print(f"  trained locally: {', '.join(local)}")
        print("  switch with FRIDAY_WAKE_MODEL=<name> (comma-separated for several)")
    return 0 if fires else 1


def _iter_wav_chunks(path: Path) -> list[bytes]:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getsampwidth() == 2
        assert w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    chunks = []
    for i in range(0, len(raw), CHUNK_SAMPLES * 2):
        piece = raw[i : i + CHUNK_SAMPLES * 2]
        if len(piece) < CHUNK_SAMPLES * 2:
            piece = piece + b"\x00" * (CHUNK_SAMPLES * 2 - len(piece))
        chunks.append(piece)
    return chunks


def run_bench(wav_path: Path = TEST_WAV, iterations: int = 20, seed: int = 0) -> list[float]:
    """Feed `wav_path` through the real detector `iterations` times with varied
    silent lead-in (shifts frame alignment) and background noise, measuring
    wall time of the single feed_chunk() call that fires detection.
    """
    rng = random.Random(seed)
    base = _iter_wav_chunks(wav_path)
    latencies_ms: list[float] = []
    for _ in range(iterations):
        detector = WakeWordDetector()
        lead_samples = rng.randint(0, CHUNK_SAMPLES - 1)
        lead = (np.random.default_rng(rng.randint(0, 2**31)).normal(0, 50, lead_samples).astype(np.int16)).tobytes()
        prefix_silence = b"\x00" * (SAMPLE_RATE)  # 1s silence before the word
        raw = prefix_silence + lead + b"".join(base)
        chunks = [raw[i : i + CHUNK_SAMPLES * 2] for i in range(0, len(raw), CHUNK_SAMPLES * 2)]
        if len(chunks[-1]) < CHUNK_SAMPLES * 2:
            chunks[-1] = chunks[-1] + b"\x00" * (CHUNK_SAMPLES * 2 - len(chunks[-1]))
        fired = False
        for chunk in chunks:
            t0 = time.perf_counter()
            detection = detector.feed_chunk(chunk)
            t1 = time.perf_counter()
            if detection is not None:
                latencies_ms.append((t1 - t0) * 1000.0)
                fired = True
                break
        if not fired:
            raise RuntimeError("bench iteration failed to detect the wake word")
    return latencies_ms


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


async def _listen_main() -> None:
    detector = WakeWordDetector()

    def _on_detection(detection: WakeDetection) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(detection.timestamp))
        print(
            f"{ts} wake={detection.model} score={detection.score:.3f} "
            f"preroll={detection.preroll_samples}samples ({detection.preroll_seconds:.2f}s)"
        )
        detector.end_handoff()

    with start_turn("reflex") as span:
        await capture_loop(detector, _on_detection, span=span)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.voice.wake")
    parser.add_argument("--listen", action="store_true", help="run live mic capture + detection")
    parser.add_argument("--bench", action="store_true", help="benchmark detection latency against test_data/alexa_test.wav")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if args.bench:
        latencies = run_bench(iterations=args.iterations)
        p50, p90, p99 = _percentile(latencies, 50), _percentile(latencies, 90), _percentile(latencies, 99)
        print(f"n={len(latencies)} p50={p50:.2f}ms p90={p90:.2f}ms p99={p99:.2f}ms")
        return 0

    if args.listen:
        asyncio.run(_listen_main())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
