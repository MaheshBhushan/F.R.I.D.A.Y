#!/usr/bin/env bash
# Generation is already complete (42,300 clips); no wait loop -- the previous
# one deadlocked because `pgrep -f generate.py` matched an unrelated shell
# whose command line merely contained that string.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-train/bin/python
for t in friday hey_friday; do
  echo "== features: $t =="
  nice -n 10 $PY features.py --target "$t" || { echo "FEATURES FAILED: $t"; exit 1; }
done
for t in friday hey_friday; do
  echo "== train: $t =="
  nice -n 10 $PY train_head.py --target "$t" || echo "TRAIN FAILED: $t"
done
echo "== install =="
$PY install_models.py || echo "INSTALL FAILED"
echo "== PIPELINE COMPLETE =="
