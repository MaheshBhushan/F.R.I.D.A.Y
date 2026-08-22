"""Synthesise positive and adversarial-negative clips with piper-sample-generator.

Runs the LibriTTS-R generator, which mixes speaker embeddings rather than
cycling a handful of fixed voices. Speaker diversity is the single biggest
driver of whether a wake word generalises to a voice it has never heard, so
the blend weights and length/noise scales are all varied per batch.

Invoked as a subprocess rather than imported: the generator brings its own
torch, its own numpy pin and a module-level model load, and running it in-process
would mean one OOM in a batch kills the whole pipeline mid-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "piper-sample-generator"
MODEL = GENERATOR / "models" / "en_US-libritts_r-medium.pt"
PYTHON = HERE / ".venv-train" / "bin" / "python"

# Speaking rates. Real commands are said faster than a TTS default, and a model
# trained only on measured speech misses the hurried "heyfriday" that a user
# actually produces once the novelty wears off.
LENGTH_SCALES = ["0.7", "0.85", "1.0", "1.15", "1.3"]
# Vocal variability. High values distort; that is the point -- clean TTS
# positives produce a model that only recognises clean TTS.
NOISE_SCALES = ["0.667", "0.8", "0.9", "1.0"]
SLERP = ["0.0", "0.25", "0.5", "0.75", "1.0"]


def generate(phrase: str, count: int, out_dir: Path, batch_size: int = 16) -> int:
    """Synthesise `count` clips of `phrase` into `out_dir`. Returns clips present.

    Resumable by design: it counts what is already there and asks only for the
    shortfall. A generation run for two phrases plus 58 adversarials takes over
    an hour, and losing all of it to one interruption is not acceptable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob("*.wav")))
    if have >= count:
        return have

    env = dict(os.environ, PYTHONPATH=str(GENERATOR))
    cmd = [
        str(PYTHON), "-m", "piper_sample_generator", phrase,
        "--max-samples", str(count - have),
        "--batch-size", str(batch_size),
        "--model", str(MODEL),
        "--output-dir", str(out_dir),
        "--length-scales", *LENGTH_SCALES,
        "--noise-scales", *NOISE_SCALES,
        "--slerp-weights", *SLERP,
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! generation failed for {phrase!r}: "
              f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '?'}",
              file=sys.stderr)
    return len(list(out_dir.glob("*.wav")))


def main() -> int:
    sys.path.insert(0, str(HERE))
    from phrases import TARGETS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "clips"))
    ap.add_argument("--positives", type=int, default=10000)
    ap.add_argument("--positives-test", type=int, default=1000)
    ap.add_argument("--adversarial", type=int, default=350,
                    help="clips per adversarial phrase")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--only", help="restrict to one target name")
    args = ap.parse_args()

    if not MODEL.exists():
        print(f"missing generator model: {MODEL}", file=sys.stderr)
        return 1

    root = Path(args.out)
    for name, spec in TARGETS.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}] positive {spec['phrase']!r}", flush=True)
        n = generate(spec["phrase"], args.positives,
                     root / name / "positive_train", args.batch_size)
        print(f"[{name}]   train {n}", flush=True)
        n = generate(spec["phrase"], args.positives_test,
                     root / name / "positive_test", args.batch_size)
        print(f"[{name}]   test  {n}", flush=True)

        for i, adv in enumerate(spec["adversarial"], 1):
            n = generate(adv, args.adversarial,
                         root / name / "adversarial" / adv.replace(" ", "_"),
                         args.batch_size)
            print(f"[{name}]   adv {i}/{len(spec['adversarial'])} "
                  f"{adv!r} -> {n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
