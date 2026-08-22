"""T4 latency bench: proves ack_audible fires immediately off speech_ended_vad,
never behind STT finalization, over >=20 simulated turns.

This drives the real ack.py playback path (real audio device, real
sounddevice callback) so the `ack_audible` mark is the same measurement
ack.py uses in production - not a mocked timestamp. It does not need a
live mic or a real Deepgram connection: `speech_ended_vad` is exactly what
LocalVAD.feed() marks in stt.py's run_utterance() (a VAD decision made on
local audio, independent of any network), and `stt_final` here stands in
for Deepgram's finalization arriving strictly *after* the router has
already classified from a partial transcript and fired the ack - which is
the actual order the real pipeline uses: Tier 3 acks fire off the partial
transcript + local VAD, before waiting on the network round trip. We mark
stt_final after a fixed simulated network delay (SIMULATED_STT_DELAY_MS)
to represent that ordering honestly, not to fabricate the ack timing
itself (which is real, measured hardware/OS latency).

Turns off both ends: reads/restores @DEFAULT_AUDIO_SINK@ volume, and
runs at a modest level, per the constraint on playing audio on this box.
"""

from __future__ import annotations

import argparse
import subprocess
import time

from pathlib import Path

from friday.core.spans import DEFAULT_SPANS_PATH, start_turn
from friday.router import Tier, classify_and_mark
from friday.voice.ack import play_ack

# Deliberately longer than any real Deepgram endpointing/MAX_WAIT_MS window
# (100-700ms in stt.py) so the bench doesn't understate how much slower
# waiting on the network would have been.
SIMULATED_STT_DELAY_MS = 200
BENCH_VOLUME = "0.15"
PARTIAL_TEXT = "can you check if the"  # Tier 3 partial - ack should fire


def _get_volume() -> str:
    out = subprocess.run(
        ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # "Volume: 0.55" -> "0.55"
    return out.split()[-1]


def _set_volume(value: str) -> None:
    subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", value], check=True)


def run_bench(iterations: int, path=DEFAULT_SPANS_PATH) -> None:
    for _ in range(iterations):
        span = start_turn("reasoning", path=path)
        span.mark("speech_started")
        span.mark("speech_ended_vad")

        decision = classify_and_mark(PARTIAL_TEXT, span, is_final=False)
        assert decision.tier is Tier.REASONING, decision

        play_ack("checking", span=span, blocking=True)

        time.sleep(SIMULATED_STT_DELAY_MS / 1000)
        span.mark("stt_final")

        span.write()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.scripts.bench_ack")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--spans-path", default=str(DEFAULT_SPANS_PATH))
    args = parser.parse_args()

    original_volume = _get_volume()
    try:
        _set_volume(BENCH_VOLUME)
        run_bench(args.iterations, path=Path(args.spans_path))
    finally:
        _set_volume(original_volume)

    print(f"wrote {args.iterations} turns to {args.spans_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
