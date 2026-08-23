#!/usr/bin/env python3
"""Validate corpus manifests and report audio-derived counts/durations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from corpus import inspect_wav


def validate(manifest: Path, root: Path) -> dict:
    counts: Counter = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    hashes: set[str] = set()
    rows = 0
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        audio = root / row["path"]
        info = inspect_wav(audio)
        if info.sha256 != row["sha256"]:
            raise ValueError(f"{manifest}:{line_number}: SHA-256 mismatch")
        if info.samples != row["samples"]:
            raise ValueError(f"{manifest}:{line_number}: sample count mismatch")
        if info.sha256 in hashes:
            raise ValueError(f"{manifest}:{line_number}: duplicate audio hash")
        hashes.add(info.sha256)
        key = f"{row['label']}:{row['split']}"
        counts[key] += 1
        durations[key] += info.duration_seconds
        rows += 1
    if not rows:
        raise ValueError(f"{manifest}: no records")
    return {
        "examples": rows,
        "counts": dict(sorted(counts.items())),
        "duration_seconds": {k: round(v, 6) for k, v in sorted(durations.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("tools/microwakeword/datasets"))
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
