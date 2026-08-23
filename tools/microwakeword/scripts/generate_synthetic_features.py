#!/usr/bin/env python3
"""Generate augmented native microWakeWord features from Piper positives."""

from __future__ import annotations

import sys
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "micro-wake-word"
sys.path.insert(0, str(UPSTREAM))

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def main() -> int:
    source = ROOT / "datasets" / "generated" / "training"
    if not next(source.glob("**/*.wav"), None):
        raise SystemExit(f"error: no Piper WAV files under {source}")
    clips = Clips(
        input_directory=str(source),
        # Only wake-only phrases are positive training labels. Longer command
        # examples remain waveform benchmarks; labeling their tail positive
        # teaches the model to wake on ordinary command words.
        file_pattern="phrase-0[0-2]/*.wav",
        remove_silence=True,
    )
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.15,
            "TanhDistortion": 0.10,
            "PitchShift": 0.15,
            "BandStopFilter": 0.10,
            "AddColorNoise": 0.35,
            "Gain": 1.0,
        },
        min_gain_db=-35,
        max_gain_db=0,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        slide_frames=10,
        step_ms=10,
    )
    target = ROOT / "datasets" / "features" / "synthetic_positives" / "training" / "friday_mmap"
    target.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=str(target),
        sample_generator=spectrograms.spectrogram_generator(repeat=2),
        batch_size=100,
        verbose=True,
    )
    print(f"source_examples={len(clips.clips)} feature_width=40 output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
