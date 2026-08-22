"""Turn synthesised WAV clips into openWakeWord feature arrays.

The feature extractor is openWakeWord's own melspectrogram + Google
speech_embedding ONNX pair, not a reimplementation. That matters more than it
sounds: the trained head is only valid for the exact features it saw, and a
subtly different mel scale or frame hop produces a model that scores well in
training and never fires in the room.

Window arithmetic, measured rather than assumed (see the shape probe in the
session log): the embedding model consumes 76 mel frames with a stride of 8, so
N embeddings need 76 + (N-1)*8 mel frames. The wake models take [1, 16, 96],
and 2.00s of 16kHz audio yields exactly 16 embeddings. Hence CLIP_SECONDS.
"""

from __future__ import annotations

import argparse
import random
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.00
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
EMBEDDINGS = 16
EMBEDDING_DIM = 96


def _read_wav(path: Path) -> "np.ndarray | None":
    """Read a mono int16 WAV, resampling to 16kHz if needed."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            rate, channels = w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())
    except (OSError, wave.Error):
        return None
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:
        # Linear resample. The generator emits 22050Hz and the feature
        # extractor is fixed at 16k; a higher-order filter is not worth it
        # here because the mel front-end discards most of what it would
        # preserve, and this runs over tens of thousands of clips.
        n_out = int(round(len(audio) * SAMPLE_RATE / rate))
        if n_out < 2:
            return None
        src = np.linspace(0.0, len(audio) - 1, n_out)
        audio = np.interp(src, np.arange(len(audio)), audio).astype(np.int16)
    return audio


def _place(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Pad/crop to exactly CLIP_SAMPLES with the phrase at a random offset.

    Random placement, not fixed: at run time the phrase can land anywhere in
    the sliding window, and a model trained with it always ending at the same
    sample learns that alignment as a feature. Measured as the difference
    between a model that fires reliably and one that fires only when you
    happen to start speaking on a window boundary.
    """
    if len(audio) >= CLIP_SAMPLES:
        start = rng.randint(0, len(audio) - CLIP_SAMPLES)
        return audio[start:start + CLIP_SAMPLES]
    room = CLIP_SAMPLES - len(audio)
    lead = rng.randint(0, room)
    return np.concatenate([
        np.zeros(lead, dtype=np.int16),
        audio,
        np.zeros(room - lead, dtype=np.int16),
    ])


def compute(paths: list, batch_size: int = 64, seed: int = 0,
            progress_every: int = 2000) -> np.ndarray:
    """Feature array of shape (n, 16, 96) for the given WAV paths."""
    from openwakeword.utils import AudioFeatures

    af = AudioFeatures()
    rng = random.Random(seed)
    out, batch, skipped = [], [], 0
    for i, path in enumerate(paths, 1):
        audio = _read_wav(Path(path))
        if audio is None or audio.size == 0:
            skipped += 1
            continue
        batch.append(_place(audio, rng))
        if len(batch) >= batch_size:
            out.append(af.embed_clips(np.stack(batch), batch_size=batch_size))
            batch = []
        if i % progress_every == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    if batch:
        out.append(af.embed_clips(np.stack(batch), batch_size=len(batch)))
    if skipped:
        print(f"    skipped {skipped} unreadable clips", flush=True)
    if not out:
        return np.zeros((0, EMBEDDINGS, EMBEDDING_DIM), dtype=np.float32)

    features = np.concatenate(out).astype(np.float32)
    # Hard invariant, not a warning. A wrong second dimension here silently
    # trains a model whose input shape does not match what the runtime feeds
    # it, and the only symptom is a wake word that never fires.
    if features.shape[1:] != (EMBEDDINGS, EMBEDDING_DIM):
        raise SystemExit(f"feature shape {features.shape} != (n, {EMBEDDINGS}, "
                         f"{EMBEDDING_DIM}); clip length is wrong")
    return features


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--out", default="features")
    ap.add_argument("--target", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    clips = Path(args.clips) / args.target
    out = Path(args.out) / args.target
    out.mkdir(parents=True, exist_ok=True)

    groups = {
        "positive_train": sorted(clips.glob("positive_train/*.wav")),
        "positive_test": sorted(clips.glob("positive_test/*.wav")),
        "adversarial": sorted(clips.glob("adversarial/*/*.wav")),
    }
    for name, paths in groups.items():
        dest = out / f"{name}.npy"
        if dest.exists():
            print(f"  {name}: exists ({np.load(dest, mmap_mode='r').shape})")
            continue
        if not paths:
            print(f"  {name}: no clips")
            continue
        print(f"  {name}: {len(paths)} clips", flush=True)
        features = compute(paths, batch_size=args.batch_size,
                           seed=abs(hash(name)) % 10_000)
        np.save(dest, features)
        print(f"  {name}: -> {features.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
