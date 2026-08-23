# Experiment artifacts

Each experiment directory retains `config.yaml`, `metrics.json`, model metadata,
model hashes, and `model.tflite` when the artifact is small enough to ship.
Large checkpoints, SavedModels, and logs are ignored and must be reproducible
from their recorded configuration and dataset manifest.

An experiment is not promotable unless `metrics.json` records successful
positive, ambient, and streaming evaluation.
