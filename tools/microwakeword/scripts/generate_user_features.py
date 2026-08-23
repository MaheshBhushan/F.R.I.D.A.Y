#!/usr/bin/env python3
"""Generate native microWakeWord features for recorded user positives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from mmap_ninja.ragged import RaggedMmap
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
UPSTREAM = ROOT / "vendor" / "micro-wake-word"
sys.path.insert(0, str(UPSTREAM))

from microwakeword.audio.audio_utils import generate_features_for_clip, remove_silence_webrtc
from microwakeword.audio.augmentation import Augmentation

WAKE_ONLY_PHRASES = {"friday", "friday?", "hey friday"}


def _rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def _generator(rows: list[dict], split: str):
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={},
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    repetitions = 10 if split == "training" else 1
    for row in rows:
        if row["split"] != split:
            continue
        if row["phrase"].strip().lower() not in WAKE_ONLY_PHRASES:
            continue
        rate, audio = wavfile.read(DATASETS / row["path"])
        assert rate == 16_000 and audio.dtype == np.int16 and audio.ndim == 1
        speech = remove_silence_webrtc(audio)
        for _ in range(repetitions):
            features = generate_features_for_clip(augmenter.augment_clip(speech), step_ms=10)
            assert features.ndim == 2 and features.shape[1] == 40
            assert features.dtype == np.float32
            if split in ("training", "validation"):
                length = features.shape[0] - 9
                for offset in range(10):
                    yield features[offset : offset + length]
            else:
                yield features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASETS / "user_positives.jsonl",
    )
    args = parser.parse_args()
    rows = _rows(args.manifest)
    counts = Counter(row["split"] for row in rows)
    durations: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        if row["phrase"].strip().lower() in WAKE_ONLY_PHRASES:
            durations[row["split"]] += row["samples"] / row["sample_rate"]

    output = DATASETS / "features" / "user_positives"
    for split in ("training", "validation", "testing"):
        wake_count = sum(
            row["split"] == split
            and row["phrase"].strip().lower() in WAKE_ONLY_PHRASES
            for row in rows
        )
        if not wake_count:
            raise ValueError(f"no {split} recordings")
        target = output / split / "friday_mmap"
        target.parent.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(target),
            sample_generator=_generator(rows, split),
            batch_size=100,
            verbose=True,
        )
        print(
            f"{split}: wake_only_examples={wake_count} "
            f"context_examples_excluded={counts[split] - wake_count} "
            f"source_duration={durations[split]:.2f}s feature_width=40"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
