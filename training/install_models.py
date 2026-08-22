"""Install trained ONNX models into the package and verify they actually load.

Verification is the point of this script existing rather than a `cp`. A wake
model that copies fine but fails to load leaves FRIDAY with zero working wake
models -- permanently deaf, with the only symptom being that she never
responds. Better to refuse the install than to ship that.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / "src" / "friday" / "models"


def verify(path: Path) -> str:
    """Load `path` through openwakeword exactly as the runtime will. Returns
    the key it registers under, which is what FRIDAY_WAKE_MODEL must name."""
    import numpy as np
    from openwakeword.model import Model

    model = Model(wakeword_models=[str(path)], inference_framework="onnx")
    keys = list(model.models.keys())
    if len(keys) != 1:
        raise SystemExit(f"{path.name}: expected 1 model, got {keys}")
    # Push real frames through: loading proves the graph parses, predicting
    # proves the input shape matches what the feature pipeline produces.
    for _ in range(20):
        scores = model.predict(np.zeros(1280, dtype=np.int16))
    if keys[0] not in scores:
        raise SystemExit(f"{path.name}: predict() returned {list(scores)}")
    return keys[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.models)
    found = sorted(src.glob("*.onnx"))
    if not found:
        print(f"no models in {src}", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    installed = []
    for path in found:
        key = verify(path)
        print(f"  {path.name}: loads, registers as {key!r}")
        if not args.dry_run:
            shutil.copy2(path, DEST / path.name)
            side = path.with_suffix(".json")
            if side.exists():
                shutil.copy2(side, DEST / side.name)
        installed.append(key)

    if args.dry_run:
        print("\ndry run; nothing copied")
        return 0
    print(f"\ninstalled into {DEST}")
    print(f"enable with:  FRIDAY_WAKE_MODEL={','.join(installed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
