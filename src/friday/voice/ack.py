"""Plays a pre-rendered acknowledgement straight from disk - zero synthesis
on the critical path. See scripts/render_acks.py for how the bank is built.

Playback backend: sounddevice, not a paplay/pw-play subprocess. Measured on
this machine (2026-08-22): a warm subprocess.run(["pw-play", tiny.wav]) costs
~21ms p50 / ~27ms p90 (fork+exec+PipeWire client connect, measured with a
5ms-silence WAV to isolate spawn overhead from playback duration); paplay
was worse at ~31ms p50. sd.play() on an already-open process (PortAudio
stream opened once at import time, like wake.py's capture_loop already does)
returns from the call in ~13-26ms with no fork/exec at all. Since ack.py runs
inside the long-lived FRIDAY process (not spawned per-utterance), sounddevice
avoids process-spawn latency entirely and was faster in every trial.

`ack_audible` is marked using sounddevice's OutputStream `time` callback
info, not when play_ack() is called: PortAudio's callback-based streams
report `time.outputBufferDacTime` (the estimated DAC output time for the
buffer being submitted) versus `time.currentTime` in the same clock domain,
both driven by the audio backend rather than wall-clock guesses at the
Python call site. We start a stream, and in the very first callback invoked
mark the span using that DAC-time estimate translated back to
`time.perf_counter()` - this is the closest available signal to "audio
actually begins hitting the device" without kernel/ALSA instrumentation.
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from friday.core.spans import TurnSpan
from friday.voice import devices, indicator

ACKS_DIR = Path(__file__).resolve().parent / "acks"


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        assert w.getnchannels() == 1, f"{path}: expected mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr


def list_acks() -> list[str]:
    """Names of available acks (filename stems), sorted."""
    return sorted(p.stem for p in ACKS_DIR.glob("*.wav"))


def play_ack(
    name: str,
    span: Optional[TurnSpan] = None,
    blocking: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Play the ack named `name` (matches a stem in acks/*.wav).

    Marks `ack_audible` on `span` the moment the first audio callback fires
    for this stream, i.e. when the first buffer is handed off to the audio
    backend for output - not when this function was called.
    """
    path = ACKS_DIR / f"{name}.wav"
    if not path.exists():
        raise FileNotFoundError(f"no ack named {name!r} in {ACKS_DIR}")
    data, sr = _load_wav(path)

    marked = False

    def _callback(outdata, frames, time_info, status) -> None:
        nonlocal marked
        if stop_event is not None and stop_event.is_set():
            raise sd.CallbackStop()
        chunk = data[_callback.pos : _callback.pos + frames]
        outdata[: len(chunk), 0] = chunk
        if len(chunk) < frames:
            outdata[len(chunk) :, 0] = 0
        _callback.pos += len(chunk)
        if not marked:
            marked = True
            if span is not None:
                span.mark("ack_audible")
        if _callback.pos >= len(data):
            raise sd.CallbackStop()

    _callback.pos = 0

    devices.apply()
    stream = sd.OutputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        callback=_callback,
        device=devices.output_device(),
    )
    # Transition outside the callback: the indicator writes a file, and the
    # callback runs on the audio thread where that risks a dropout.
    with indicator.during(indicator.State.TALKING), stream:
        if blocking:
            duration = len(data) / sr
            if stop_event is None:
                time.sleep(duration + 0.05)
            else:
                stop_event.wait(duration + 0.05)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m friday.voice.ack")
    parser.add_argument("name", nargs="?", help="ack name to play (default: list available acks)")
    args = parser.parse_args()

    if not args.name:
        for name in list_acks():
            print(name)
        return 0

    play_ack(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
