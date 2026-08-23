"""Small, dependency-free corpus manifest helpers."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class WavInfo:
    samples: int
    sample_rate: int
    channels: int
    sample_width: int
    duration_seconds: float
    sha256: str


def inspect_wav(path: Path) -> WavInfo:
    """Validate FRIDAY's canonical WAV format and derive duration from audio."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        samples = source.getnframes()
    if (sample_rate, channels, sample_width) != (
        SAMPLE_RATE,
        CHANNELS,
        SAMPLE_WIDTH,
    ):
        raise ValueError(
            f"{path}: expected 16 kHz mono 16-bit PCM, got "
            f"{sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit"
        )
    return WavInfo(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_seconds=samples / sample_rate,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def split_for_session(session: str, seed: int = 20260823) -> str:
    """Assign whole sessions deterministically, preventing speaker leakage."""
    digest = hashlib.sha256(f"{seed}:{session}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10
    if bucket == 8:
        return "validation"
    if bucket == 9:
        return "testing"
    return "training"


def manifest_record(path: Path, *, root: Path, **metadata: object) -> dict:
    info = inspect_wav(path)
    return {
        "path": path.relative_to(root).as_posix(),
        **asdict(info),
        **metadata,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, sort_keys=True) + "\n")
