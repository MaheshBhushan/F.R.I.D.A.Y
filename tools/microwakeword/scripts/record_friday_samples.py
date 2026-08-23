#!/usr/bin/env python3
"""Record primary-user positives through FRIDAY's single-owner audio path."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import wave
from itertools import cycle
from pathlib import Path

from friday.audio.capture import FRAME_MS, AudioCaptureService
from friday.audio.manager import AudioResourceManager

from corpus import append_jsonl, manifest_record, split_for_session

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

PROMPTS = (
    "Friday",
    "Friday?",
    "Hey Friday",
    "Friday, yes",
    "Friday, no",
    "Friday, stop",
    "Friday, wait",
    "Friday, hello",
    "Friday, are you working?",
    "Friday, check Codex",
    "Friday, what's happening?",
    "Friday, what's Codex doing?",
)

CONDITIONS = (
    "normal-close",
    "normal-medium",
    "normal-far",
    "quiet-close",
    "quiet-medium",
    "loud-medium",
    "fast",
    "slow",
    "question-intonation",
    "standing",
    "head-turned",
    "room-noise",
)


def _friday_running() -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "friday.service"],
        check=False,
    )
    return result.returncode == 0


async def _record(path: Path, duration_seconds: float) -> None:
    frames_needed = round(duration_seconds * 1000 / FRAME_MS)
    manager = AudioResourceManager(manage_echo_cancel=False)
    capture = AudioCaptureService(manager)
    stop = asyncio.Event()
    subscription = capture.subscribe_live()
    await manager.start()
    task = asyncio.create_task(capture.run(stop))
    pcm: list[bytes] = []
    try:
        async for frame in subscription:
            pcm.append(frame.pcm)
            if len(pcm) >= frames_needed:
                break
    finally:
        subscription.close()
        stop.set()
        await asyncio.gather(task, return_exceptions=True)
        await manager.stop()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"".join(pcm))


async def _main(args: argparse.Namespace) -> int:
    if _friday_running():
        print(
            "error: stop FRIDAY first so AudioCaptureService remains the only "
            "microphone owner:\n  friday stop",
            file=sys.stderr,
        )
        return 2

    session_dir = DATASETS / "raw" / "user" / args.session
    manifest = DATASETS / "user_positives.jsonl"
    prompt_cycle = cycle(PROMPTS)
    condition_cycle = cycle(CONDITIONS)
    existing = len(list(session_dir.glob("*.wav"))) if session_dir.exists() else 0

    print(f"session={args.session} split={split_for_session(args.session, args.seed)}")
    print("Press Enter for each take, then speak when 'recording' appears. Ctrl-C stops.")
    for offset in range(args.count):
        index = existing + offset + 1
        prompt = next(prompt_cycle)
        condition = next(condition_cycle)
        input(f"\n[{index:04d}] {condition} — say: {prompt!r}  ")
        print("recording...", flush=True)
        output = session_dir / f"{index:04d}.wav"
        await _record(output, args.duration)
        record = manifest_record(
            output,
            root=DATASETS,
            label="positive",
            split=split_for_session(args.session, args.seed),
            source="primary-user",
            session=args.session,
            license="private-user-data",
            phrase=prompt,
            condition=condition,
        )
        append_jsonl(manifest, record)
        print(f"saved {output} ({record['duration_seconds']:.2f}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="stable recording-session ID")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.count < 1 or args.duration <= 0:
        parser.error("--count and --duration must be positive")
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nstopped")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
