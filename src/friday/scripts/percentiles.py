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


def print_table(records: list[dict]) -> None:
    stage_values: dict[str, list[int]] = {}
    for rec in records:
        for stage, offset in rec.get("stages", {}).items():
            stage_values.setdefault(stage, []).append(offset)

    print(f"{'stage':<24} {'count':>6} {'p50(ms)':>10} {'p90(ms)':>10} {'p99(ms)':>10}")
    for stage in sorted(stage_values):
        values = sorted(stage_values[stage])
        p50 = pct(values, 50) / 1e6
        p90 = pct(values, 90) / 1e6
        p99 = pct(values, 99) / 1e6
        print(f"{stage:<24} {len(values):>6} {p50:>10.3f} {p90:>10.3f} {p99:>10.3f}")

    print()
    print("derived deltas:")
    print(f"{'delta':<40} {'count':>6} {'p50(ms)':>10} {'p90(ms)':>10} {'p99(ms)':>10}")
    for label, a, b in DELTAS:
        values = []
        for rec in records:
            stages = rec.get("stages", {})
            if a in stages and b in stages:
                values.append(stages[a] - stages[b])
        if not values:
            print(f"{label:<40} {0:>6} {'--':>10} {'--':>10} {'--':>10}")
            continue
        values.sort()
        p50 = pct(values, 50) / 1e6
        p90 = pct(values, 90) / 1e6
        p99 = pct(values, 99) / 1e6
        print(f"{label:<40} {len(values):>6} {p50:>10.3f} {p90:>10.3f} {p99:>10.3f}")


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
