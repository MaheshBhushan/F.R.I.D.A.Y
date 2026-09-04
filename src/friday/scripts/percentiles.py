"""Print per-stage p50/p90/p99 latency table (and key deltas) from spans.jsonl.

Percentiles only - no mean/average, since averages hide the tail.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_SPANS_PATH = Path.home() / ".friday" / "spans.jsonl"

DELTAS = (
    ("ack_audible - speech_ended_vad", "ack_audible", "speech_ended_vad"),
    ("stt_final - speech_ended_vad", "stt_final", "speech_ended_vad"),
    ("first_token - llm_sent", "first_token", "llm_sent"),
    ("first_content_audio - speech_ended_vad", "first_content_audio", "speech_ended_vad"),
    ("task_complete - speech_ended_vad", "task_complete", "speech_ended_vad"),
)


def load_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pct(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[q - 1]


def summarize(records: list[dict]) -> dict:
    """Per-stage and per-delta percentiles as data. Percentiles only."""
    stage_values: dict[str, list[int]] = {}
    for rec in records:
        for stage, offset in rec.get("stages", {}).items():
            stage_values.setdefault(stage, []).append(offset)
    stages = []
    for stage in sorted(stage_values):
        values = sorted(stage_values[stage])
        stages.append(_row(stage, values))
    deltas = []
    for label, a, b in DELTAS:
        values = sorted(
            rec["stages"][a] - rec["stages"][b]
            for rec in records
            if a in rec.get("stages", {}) and b in rec.get("stages", {})
        )
        deltas.append(_row(label, values))
    kinds = {}
    for rec in records:
        kinds[rec.get("turn_kind", "?")] = kinds.get(rec.get("turn_kind", "?"), 0) + 1
    return {"turns": len(records), "kinds": kinds, "stages": stages, "deltas": deltas}


def _row(name: str, values: list[int]) -> dict:
    if not values:
        return {"name": name, "count": 0, "p50_ms": None, "p90_ms": None, "p99_ms": None}
    return {
        "name": name,
        "count": len(values),
        "p50_ms": round(pct(values, 50) / 1e6, 3),
        "p90_ms": round(pct(values, 90) / 1e6, 3),
        "p99_ms": round(pct(values, 99) / 1e6, 3),
    }


def filter_records(
    records: list[dict], *, kind: str | None = None, last: int | None = None
) -> list[dict]:
    if kind:
        records = [r for r in records if r.get("turn_kind") == kind]
    if last:
        records = records[-last:]
    return records


def format_table(summary: dict) -> str:
    def fmt(v):
        return "--" if v is None else f"{v:.3f}"

    kinds = ", ".join(f"{k}={n}" for k, n in sorted(summary["kinds"].items()))
    lines = [f"{summary['turns']} turns ({kinds})", ""]
    lines.append(f"{'stage':<40} {'count':>6} {'p50(ms)':>10} {'p90(ms)':>10} {'p99(ms)':>10}")
    for row in summary["stages"]:
        lines.append(
            f"{row['name']:<40} {row['count']:>6} {fmt(row['p50_ms']):>10} "
            f"{fmt(row['p90_ms']):>10} {fmt(row['p99_ms']):>10}"
        )
    lines += ["", "derived deltas:"]
    lines.append(f"{'delta':<40} {'count':>6} {'p50(ms)':>10} {'p90(ms)':>10} {'p99(ms)':>10}")
    for row in summary["deltas"]:
        lines.append(
            f"{row['name']:<40} {row['count']:>6} {fmt(row['p50_ms']):>10} "
            f"{fmt(row['p90_ms']):>10} {fmt(row['p99_ms']):>10}"
        )
    return "\n".join(lines)


def print_table(records: list[dict]) -> None:
    print(format_table(summarize(records)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(DEFAULT_SPANS_PATH))
    args = parser.parse_args(argv)

    records = load_records(Path(args.path))
    if not records:
        print(f"no records found at {args.path}")
        return 0
    print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
