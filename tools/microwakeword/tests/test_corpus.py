from __future__ import annotations

import sys
import wave
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from corpus import inspect_wav, split_for_session  # noqa: E402


def _wav(path: Path, samples: int) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * samples)


def test_duration_comes_from_pcm_samples_not_feature_rows(tmp_path):
    path = tmp_path / "one-second.wav"
    _wav(path, 16_000)
    info = inspect_wav(path)
    assert info.samples == 16_000
    assert info.duration_seconds == 1.0
    # A 10 ms feature pipeline would produce roughly 100 overlapping rows;
    # treating each row as a full analysis window recreates the old bug.
    feature_rows = 100
    assert info.duration_seconds != feature_rows * 0.16


def test_rejects_wrong_audio_layout(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * 32_000)
    try:
        inspect_wav(path)
    except ValueError as exc:
        assert "mono" in str(exc)
    else:
        raise AssertionError("invalid layout was accepted")


def test_session_split_is_deterministic_and_grouped():
    first = split_for_session("session-2026-08-23")
    assert first == split_for_session("session-2026-08-23")
    assert first in {"training", "validation", "testing"}
