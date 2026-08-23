"""End-to-end gate: does the exported model fire through the real runtime?

This exists because every other check in this pipeline validated the head
against the same features that produced it. A model trained on `embed_clips`
output scored 0.942 recall, passed the recall floor, passed the false-accept
budget, exported, installed, and then fired on 0 of 20 of its own training
positives through `openwakeword.Model.predict` -- the path FRIDAY actually
uses. Offline metrics cannot catch a train/serve skew by construction.

So this loads the .onnx exactly as the daemon does and plays audio at it.
Nothing ships unless this passes.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHUNK = 1280
LEAD_SILENCE = 32000    # 2s, so the model has a full window of context before
                        # the phrase, as it would mid-conversation


def _stream_score(model, audio: np.ndarray) -> float:
    """Peak score over a clip, fed frame by frame like a live microphone."""
    model.reset()
    stream = np.concatenate([
        np.zeros(LEAD_SILENCE, dtype=np.int16),
        audio,
        np.zeros(LEAD_SILENCE, dtype=np.int16),
    ])
    best = 0.0
    for i in range(0, len(stream) - CHUNK, CHUNK):
        scores = model.predict(stream[i:i + CHUNK])
        best = max(best, max(scores.values()))
    return best


def main() -> int:
    import features as F
    import openwakeword

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--min-recall", type=float, default=0.60,
                    help="real-path recall below which the model does not ship")
    args = ap.parse_args()

    model_path = args.model or f"models/{args.target}.onnx"
    if not Path(model_path).is_file():
        print(f"no model at {model_path}", file=sys.stderr)
        return 1

    model = openwakeword.Model(wakeword_models=[model_path],
                               inference_framework="onnx")

    def score_dir(pattern: str, limit: int) -> "list[float]":
        out = []
        for p in sorted(glob.glob(pattern))[:limit]:
            audio = F._read_wav(Path(p))
            if audio is None or audio.size == 0:
                continue
            out.append(_stream_score(model, audio))
        return out

    root = Path(args.clips) / args.target
    pos = score_dir(str(root / "positive_test" / "*.wav"), args.samples)
    adv = score_dir(str(root / "adversarial" / "*" / "*.wav"), args.samples)

    if not pos:
        print("no held-out positives to verify against", file=sys.stderr)
        return 1

    pos = np.array(pos)
    recall = float((pos >= args.threshold).mean())
    print(f"\n{args.target}: real-path verification via Model.predict")
    print(f"  positives   {len(pos):>4}  recall {recall:.3f}  "
          f"median score {np.median(pos):.3f}")
    if adv:
        adv = np.array(adv)
        fire = float((adv >= args.threshold).mean())
        print(f"  adversarial {len(adv):>4}  fire rate {fire:.3f}  "
              f"median score {np.median(adv):.3f}")

    if recall < args.min_recall:
        print(f"\nFAILED: real-path recall {recall:.3f} < {args.min_recall:.2f}. "
              f"The model does not work through the runtime regardless of what "
              f"the offline metrics said.", file=sys.stderr)
        return 1
    print(f"\nPASSED: fires through the real runtime "
          f"({recall:.1%} of held-out positives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
