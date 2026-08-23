# FRIDAY microWakeWord tooling

This directory is an isolated training and evaluation workspace. It does not
change FRIDAY's production wake backend and does not install TensorFlow or
PyTorch into the project's main `.venv`.

The canonical upstream revisions and design are recorded in
[`docs/microwakeword_integration_plan.md`](../../docs/microwakeword_integration_plan.md).

## Environment

Use Python 3.10, as recommended by the upstream training notebook:

```bash
cd tools/microwakeword
uv venv --python 3.10 .venv
uv pip sync --torch-backend cpu --python .venv/bin/python requirements.lock
```

`requirements.txt` contains the human-maintained top-level requirements;
`requirements.lock` is the fully resolved Python 3.10 environment. Regenerate
it with:

```bash
uv pip compile --python-version 3.10 --torch-backend cpu requirements.txt \
  -o requirements.lock
```

The CPU backend is deliberate: NANI has Intel graphics and resolving PyTorch's
default Linux wheels downloads several gigabytes of unusable NVIDIA libraries.

Do not run `uv sync` from this directory: the repository root and this training
workspace intentionally have different dependency sets.

## Data policy

Only corpus manifests, hashes, licenses, configurations, and benchmark result
JSON belong in git. Raw user recordings, third-party audio, generated samples,
Ragged Mmap features, checkpoints, and training caches are ignored.

Every source manifest must record its license and provenance. The convenient
mixed datasets from the upstream notebook are personal/non-commercial unless
each source's terms have been reviewed independently.

## Planned workflow

```text
record/generate WAV
  -> validate 16 kHz mono int16 and hash source
  -> grouped deterministic split
  -> native microWakeWord augmentation and feature generation
  -> train several candidates
  -> quantized internal-state TFLite export
  -> byte-identical OpenWakeWord/microWakeWord benchmark
  -> live acceptance benchmark
  -> recommendation; no automatic backend switch
```

The required human-data gate is at least 100 primary-user recordings with
separate held-out sessions. A candidate cannot be promoted without real
ambient false-accept measurements.
