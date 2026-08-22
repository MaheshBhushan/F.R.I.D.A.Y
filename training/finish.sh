#!/usr/bin/env bash
# Wait out generation, then run features -> train -> install unattended.
# One job, one notification: polling this from the agent side costs tokens per
# check, and there is nothing to decide between stages anyway.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-train/bin/python

while pgrep -f "generate.py" >/dev/null; do sleep 60; done
echo "== generation done: $(find clips -name '*.wav' | wc -l) clips =="

for t in friday hey_friday; do
  echo "== features: $t =="
  $PY features.py --target "$t" || { echo "FEATURES FAILED: $t"; exit 1; }
done

for t in friday hey_friday; do
  echo "== train: $t =="
  $PY train_head.py --target "$t" || echo "TRAIN FAILED: $t"
done

echo "== install =="
$PY install_models.py || echo "INSTALL FAILED"
echo "== PIPELINE COMPLETE =="
