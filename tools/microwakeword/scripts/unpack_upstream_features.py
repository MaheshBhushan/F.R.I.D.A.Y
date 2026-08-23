#!/usr/bin/env python3
"""Safely unpack canonical precomputed negative feature archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "datasets" / "downloads"
FEATURES = ROOT / "datasets" / "features" / "upstream"
ARCHIVES = ("dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip")


def main() -> int:
    FEATURES.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        source = DOWNLOADS / name
        if not source.is_file():
            raise SystemExit(f"error: missing {source}")
        print(f"unpacking {name}", flush=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (FEATURES / member.filename).resolve()
                if not target.is_relative_to(FEATURES.resolve()):
                    raise ValueError(f"unsafe archive member: {member.filename}")
            archive.extractall(FEATURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
