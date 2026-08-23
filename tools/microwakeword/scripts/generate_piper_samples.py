#!/usr/bin/env python3
"""Generate labeled Friday positives with the pinned Piper checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "2971426a55072f7d22fec416ca7800df8bd23207"
DEFAULT_PHRASES = (
    "Friday",
    "Friday?",
    "Hey Friday",
    "Friday, are you working?",
    "Friday, check Codex.",
    "Friday, stop.",
    "Friday, what's happening?",
)


def _checkout() -> Path:
    path = ROOT / "vendor" / "piper-sample-generator"
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or result.stdout.strip() != PINNED_COMMIT:
        raise SystemExit(
            "error: missing/wrong pinned Piper checkout; follow tools/"
            "microwakeword/README.md"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples-per-phrase", type=int, default=500)
    parser.add_argument("--split", choices=("training", "validation", "testing"), default="training")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--phrases-json", type=Path)
    args = parser.parse_args()
    if args.samples_per_phrase < 1:
        parser.error("--samples-per-phrase must be positive")

    checkout = _checkout()
    model = args.model.resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    phrases = DEFAULT_PHRASES
    if args.phrases_json:
        phrases = tuple(json.loads(args.phrases_json.read_text()))

    output_root = ROOT / "datasets" / "generated" / args.split
    for index, phrase in enumerate(phrases):
        output = output_root / f"phrase-{index:02d}"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "piper_sample_generator",
            phrase,
            "--model",
            str(model),
            "--max-samples",
            str(args.samples_per_phrase),
            "--batch-size",
            str(args.batch_size),
            "--max-speakers",
            "800",
            "--length-scales",
            "0.75",
            "0.9",
            "1.0",
            "1.15",
            "1.3",
            "--noise-scales",
            "0.5",
            "0.667",
            "0.8",
            "--noise-scale-ws",
            "0.6",
            "0.8",
            "1.0",
            "--slerp-weights",
            "0.0",
            "0.25",
            "0.5",
            "0.75",
            "1.0",
            "--output-dir",
            str(output),
        ]
        print(f"[{index + 1}/{len(phrases)}] {phrase!r}", flush=True)
        subprocess.run(command, cwd=checkout, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
