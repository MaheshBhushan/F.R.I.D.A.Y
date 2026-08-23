#!/usr/bin/env python3
"""Run one pinned upstream training experiment and record its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
UPSTREAM = ROOT / "vendor" / "micro-wake-word"
UPSTREAM_COMMIT = "4665173cd35f1cff9a61e06fc427f124766c488e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="NVMe/local cache replacing datasets/features for training I/O",
    )
    args = parser.parse_args()
    if _git("rev-parse", "HEAD", cwd=UPSTREAM) != UPSTREAM_COMMIT:
        raise SystemExit("error: canonical microWakeWord checkout is not pinned")

    artifact = ROOT / "artifacts" / args.experiment
    artifact.mkdir(parents=True, exist_ok=True)
    config = args.config.resolve()
    shutil.copy2(config, artifact / "config.yaml")
    runtime_config = yaml.safe_load(config.read_text())
    if args.feature_root:
        source_root = (ROOT / "datasets" / "features").resolve()
        feature_root = args.feature_root.resolve()
        for feature_set in runtime_config["features"]:
            source = (REPO / feature_set["features_dir"]).resolve()
            try:
                relative = source.relative_to(source_root)
            except ValueError as exc:
                raise SystemExit(f"error: feature path is outside source root: {source}") from exc
            cached = feature_root / relative
            if not cached.is_dir():
                raise SystemExit(f"error: cached feature directory is missing: {cached}")
            feature_set["features_dir"] = str(cached)
    runtime_config_path = artifact / "runtime-config.yaml"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False))
    manifest = ROOT / "datasets" / "user_positives.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    metadata = {
        "experiment_id": args.experiment,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "upstream_commit": UPSTREAM_COMMIT,
        "training_config_sha256": _sha(config),
        "dataset_manifest_sha256": _sha(manifest),
        "feature_root": str(args.feature_root.resolve()) if args.feature_root else None,
        "real_positive_count": len(rows),
        "real_positive_duration_seconds": sum(row["samples"] / row["sample_rate"] for row in rows),
    }
    metrics_path = artifact / "metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={runtime_config_path}",
        "--train=1",
        "--restore_checkpoint=1",
        "--test_tf_nonstreaming=0",
        "--test_tflite_nonstreaming=0",
        "--test_tflite_nonstreaming_quantized=0",
        "--test_tflite_streaming=0",
        "--test_tflite_streaming_quantized=1",
        "--use_weights=best_weights",
        "mixednet",
        "--pointwise_filters=64,64,64,64",
        "--repeat_in_block=1,1,1,1",
        "--mixconv_kernel_sizes=[5],[7,11],[9,15],[23]",
        "--residual_connection=0,0,0,0",
        "--first_conv_filters=32",
        "--first_conv_kernel_size=5",
        "--stride=3",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(UPSTREAM)
    log_path = artifact / "training.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
        if code:
            raise RuntimeError(f"training exited {code}")

        work = REPO / "tools" / "microwakeword" / "artifacts" / args.experiment / "work"
        model = work / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
        if not model.is_file():
            raise RuntimeError(f"training succeeded without expected model: {model}")
        target = artifact / "model.tflite"
        shutil.copy2(model, target)
        metadata.update(
            status="trained",
            completed_at=datetime.now(timezone.utc).isoformat(),
            model_sha256=_sha(target),
            model_size_bytes=target.stat().st_size,
        )
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc))
        metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        raise
    metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
