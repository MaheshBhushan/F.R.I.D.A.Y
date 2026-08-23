#!/usr/bin/env bash
# Full rebuild after the train/serve skew fix: extract streaming-path features,
# retrain, and gate on the REAL inference path before installing anything.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-train/bin/python
TARGETS="friday hey_friday"

for t in $TARGETS; do
  echo "== features (streaming): $t =="
  nice -n 10 $PY features.py --target "$t" --mode both \
    || { echo "FEATURES FAILED: $t"; exit 1; }
done

for t in $TARGETS; do
  echo "== train: $t =="
  nice -n 10 $PY train_head.py --target "$t" || { echo "TRAIN FAILED: $t"; continue; }
  echo "== verify: $t =="
  nice -n 10 $PY verify.py --target "$t" || { echo "VERIFY FAILED: $t"; exit 1; }
done

echo "== install =="
$PY install_models.py || { echo "INSTALL FAILED"; exit 1; }
echo "== REBUILD COMPLETE =="
