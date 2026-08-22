#!/usr/bin/env bash
# Full wake-word training pipeline, resumable at every stage.
#
# Each stage skips work that already exists, so re-running after an
# interruption costs only what was lost. Total from cold: ~3h on this
# 8-core CPU, dominated by TTS generation.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-train/bin/python
TARGETS="${TARGETS:-friday hey_friday}"

echo "== 1/4 generate clips =="
$PY generate.py

echo "== 2/4 compute features =="
for t in $TARGETS; do echo "[$t]"; $PY features.py --target "$t"; done

echo "== 3/4 train =="
for t in $TARGETS; do echo "[$t]"; $PY train_head.py --target "$t"; done

echo "== 4/4 install =="
$PY install_models.py
