"""Offline renderer for the pre-rendered acknowledgement bank (T4).

This is a BUILD-TIME script, never run on the critical path. It produces
src/friday/voice/acks/*.wav — short "Checking.", "On it." style phrases
that ack.py plays straight from disk with zero synthesis at request time.

Two engines:
  - "aura": Deepgram Aura TTS, the intended production voice. Requires
    DEEPGRAM_API_KEY. Not available yet (see project brief) - calling
    this engine without the key exits with a clear, actionable message.
  - "espeak-ng": offline fallback so the bank exists and is measurable
    today. espeak-ng is a system package (not a Python/ML dependency);
    installed via `pacman -S espeak-ng` on this machine, since neither
    it nor pico2wave was present. Its native output is 22050Hz mono
    16-bit PCM; we resample to TARGET_SAMPLE_RATE via ffmpeg (also
    already on this machine) to match Aura's default WAV output format
    (mono 16-bit linear PCM, 24000Hz) so the bank's format doesn't
    change shape when the real key arrives.

Every run writes a SOURCE marker file into the bank directory recording
which engine produced the files, so it's unambiguous whether the bank
is live (aura) or a stand-in (espeak-ng). Re-running with --engine aura
overwrites placeholder files with real Aura audio.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ACKS_DIR = Path(__file__).resolve().parent.parent / "voice" / "acks"
TARGET_SAMPLE_RATE = 24000  # matches Deepgram Aura's default linear16 WAV output

# name -> (filename stem, phrase to render)
PHRASES: dict[str, str] = {
    "checking": "Checking.",
    "on_it": "On it.",
    "got_it": "Got it.",
    "found_it": "Found it.",
    "sir": "Sir?",
    "yes": "Yes?",
    "done": "Done.",
    "working_on_it": "Working on it.",
    "one_moment": "One moment.",
    "sure": "Sure.",
    "right_away": "Right away.",
    "looking_into_it": "Looking into it.",
    "okay": "Okay.",
    "give_me_a_second": "Give me a second.",
    "understood": "Understood.",
}


def _require_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        print(
            "error: DEEPGRAM_API_KEY is not set. Aura rendering needs it - "
            "export DEEPGRAM_API_KEY=<your key> and retry with --engine aura. "
            "Until then, use --engine espeak-ng for a placeholder bank.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key


def _render_espeak_ng(phrase: str, out_path: Path) -> None:
    """Render `phrase` to a mono 16-bit PCM WAV at TARGET_SAMPLE_RATE via
    espeak-ng (native output, 22050Hz) piped through ffmpeg for resample."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["espeak-ng", "-w", str(tmp_path), "-s", "175", phrase],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(tmp_path),
                "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
                str(out_path),
            ],
            check=True, capture_output=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _render_aura(phrase: str, out_path: Path, api_key: str, model: str) -> None:
    """Render `phrase` via Deepgram Aura's REST TTS endpoint to a WAV file
    at TARGET_SAMPLE_RATE, mono, linear16."""
    import httpx

    url = f"https://api.deepgram.com/v1/speak?model={model}&encoding=linear16&sample_rate={TARGET_SAMPLE_RATE}&container=wav"
    resp = httpx.post(
        url,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        json={"text": phrase},
        timeout=30.0,
    )
    resp.raise_for_status()
    _write_normalized_wav(out_path, resp.content)


def _write_normalized_wav(out_path: Path, payload: bytes) -> None:
    """Rewrite a streamed WAV with correct RIFF/data sizes.

    Aura streams the container, so it declares placeholder sizes (data size
    0x7fff0000). The samples are fine but `wave` then reports ~1.07e9 frames,
    and any consumer deriving a duration from the header -- ack.py sizes its
    blocking sleep that way -- would wait ~12 hours per ack.
    """
    idx = payload.find(b"data")
    pcm = payload[idx + 8:] if idx != -1 else payload
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SAMPLE_RATE)
        w.writeframes(pcm)


def _duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def render_bank(engine: str, aura_model: str = "aura-2-thalia-en") -> None:
    ACKS_DIR.mkdir(parents=True, exist_ok=True)
    api_key = _require_api_key() if engine == "aura" else None

    for stem, phrase in PHRASES.items():
        out_path = ACKS_DIR / f"{stem}.wav"
        if engine == "aura":
            _render_aura(phrase, out_path, api_key, aura_model)
        else:
            _render_espeak_ng(phrase, out_path)
        dur = _duration_seconds(out_path)
        print(f"{out_path.name:<24} {dur:.2f}s  ({phrase!r})")

    source_marker = (
        f"deepgram aura (model={aura_model}, {TARGET_SAMPLE_RATE}Hz mono s16)"
    ) if engine == "aura" else f"espeak-ng (placeholder, model=default, {TARGET_SAMPLE_RATE}Hz mono s16)"
    (ACKS_DIR / "SOURCE").write_text(source_marker + "\n", encoding="utf-8")
    print(f"\nwrote {len(PHRASES)} files to {ACKS_DIR}, SOURCE={source_marker}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.scripts.render_acks")
    parser.add_argument(
        "--engine", choices=("aura", "espeak-ng"), default="aura",
        help="TTS engine to render the bank with (default: aura, the production voice)",
    )
    parser.add_argument("--aura-model", default="aura-2-thalia-en", help="Deepgram Aura voice model")
    args = parser.parse_args()

    render_bank(args.engine, aura_model=args.aura_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
