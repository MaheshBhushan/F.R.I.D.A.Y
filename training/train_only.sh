#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-train/bin/python
for t in friday hey_friday; do
  echo "== train: $t =="
  nice -n 10 $PY train_head.py --target "$t" || echo "TRAIN FAILED: $t"
done
echo "== install =="
$PY install_models.py || echo "INSTALL FAILED"
echo "== ALL DONE =="
