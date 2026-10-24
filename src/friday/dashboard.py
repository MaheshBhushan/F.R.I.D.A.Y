"""Local runtime dashboard and alert monitor over FRIDAY's existing spans."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template

from friday.core import events
from friday.core.spans import DEFAULT_SPANS_PATH, active_spans_dir
from friday.scripts.percentiles import summarize


def _nonnegative(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _valid(record) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("turn_id"), str) and bool(record["turn_id"])
        and isinstance(record.get("turn_kind"), str)
        and _nonnegative(record.get("wall_start"))
        and isinstance(record.get("stages"), dict)
        and all(_nonnegative(v) for v in record["stages"].values())
        and isinstance(record.get("durations_ns", {}), dict)
        and all(_nonnegative(v) for v in record.get("durations_ns", {}).values())
    )


def read_traces(path: Path, limit: int) -> tuple[list[dict], int]:
    """Read active snapshots first; final records take precedence during cleanup."""
    records = {}
    invalid = 0

    def accept(line):
        nonlocal invalid
        if not line.strip():
            return
        try:
            record = json.loads(line)
        except (ValueError, RecursionError):
            invalid += 1
            return
        if _valid(record):
            records[record["turn_id"]] = record
        else:
            invalid += 1

    for checkpoint in active_spans_dir(path).glob("*.json"):
        try:
            accept(checkpoint.read_text(encoding="utf-8"))
        except FileNotFoundError:  # Final write may remove it after glob().
            continue

    # ponytail: stream the log on each poll; add an offset index if scans grow slow.
    try:
        with path.open(encoding="utf-8") as source:
            for line in deque(source, maxlen=limit):
                accept(line)
    except FileNotFoundError:
        pass
    return sorted(records.values(), key=lambda r: r["wall_start"])[-limit:], invalid


def evaluate_alerts(records: list[dict], *, now: float,
                    slow_ms: float, missing_ms: float) -> list[dict]:
    alerts = []
    for record in records:
        stages = record["stages"]
        # Wake-only and STT-only traces do not establish a response obligation.
        if not any(s in stages for s in ("intent_classified", "llm_sent", "tts_started",
                                         "first_content_audio", "task_complete")):
            continue
        start = next((stages[s] for s in ("speech_ended_vad", "stt_final",
                      "intent_classified", "llm_sent", "tts_started") if s in stages), 0)
        response = stages.get("first_content_audio")
        if response is None and record["turn_kind"] == "reflex":
            response = stages.get("ack_audible")
        if response is None:
            response = stages.get("task_complete")  # Text-only turns have no audio.
        code = None
        if response is None:
            elapsed_ms = max(0, (now - record["wall_start"]) * 1000 - start / 1e6)
            # Compare absolute deadlines to avoid cancellation at epoch-sized clocks.
            if now >= record["wall_start"] + start / 1e9 + missing_ms / 1000:
                code = "missing_response"
        else:
            elapsed_ms = max(0, (response - start) / 1e6)
            if elapsed_ms > slow_ms:
                code = "slow_response"
        if code:
            alerts.append({"turn_id": record["turn_id"], "turn_kind": record["turn_kind"],
                           "code": code, "elapsed_ms": round(elapsed_ms, 3)})
    return alerts


def snapshot(path: Path, *, slow_ms: float, missing_ms: float,
             limit: int, now: float | None = None) -> dict:
    error = None
    try:
        records, invalid = read_traces(path, limit)
    except (OSError, UnicodeError) as exc:
        records, invalid, error = [], 0, str(exc)
    return {
        "summary": summarize(records), "records": records, "invalid_records": invalid,
        "source_error": error,
        "thresholds": {"slow_ms": slow_ms, "missing_ms": missing_ms, "limit": limit},
        "alerts": evaluate_alerts(records, now=time.time() if now is None else now,
                                  slow_ms=slow_ms, missing_ms=missing_ms),
    }


def create_app(path: Path = DEFAULT_SPANS_PATH, *, slow_ms: float = 3000,
               missing_ms: float = 10000, limit: int = 200) -> Flask:
    """Create a read-only app; importing it starts no server or monitor."""
    if not all(_nonnegative(v) and v > 0 for v in (slow_ms, missing_ms)):
        raise ValueError("alert thresholds must be positive finite milliseconds")
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("limit must be an integer from 1 to 10000")
    app = Flask(__name__)
    path = Path(path).expanduser()

    def collect():
        return snapshot(path, slow_ms=slow_ms, missing_ms=missing_ms, limit=limit)

    app.extensions["friday_snapshot"] = collect

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def dashboard():
        data = collect()
        return render_template("dashboard.html", data=data), 503 if data["source_error"] else 200

    @app.get("/api/metrics")
    def metrics():
        data = collect()
        return jsonify(data), 503 if data["source_error"] else 200

    return app


class AlertMonitor:
    """Emit one local event per alert transition, independent of HTTP requests."""

    def __init__(self, collect):
        self.collect = collect
        self.active: dict[tuple[str, str], dict] = {}

    def poll(self) -> None:
        data = self.collect()
        current = {(a["turn_id"], a["code"]): a for a in data["alerts"]}
        if data["source_error"]:
            current = dict(self.active)  # A read failure does not prove recovery.
            current[("source", "source_error")] = {"code": "source_error",
                                                     "error": data["source_error"]}
        for key, alert in current.items():
            if key not in self.active:
                events.emit("alert", **alert)
        for key in self.active.keys() - current.keys():
            events.emit("alert-cleared", turn_id=key[0], code=key[1])
        self.active = current

    def run(self, stop: threading.Event, interval: float) -> None:
        while not stop.is_set():
            self.poll()
            stop.wait(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_SPANS_PATH)
    parser.add_argument("--slow-ms", type=float, default=3000)
    parser.add_argument("--missing-ms", type=float, default=10000)
    parser.add_argument("--last", type=int, default=200)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--check", action="store_true", help="print a replay snapshot and exit; 1 if alerts")
    args = parser.parse_args(argv)
    try:
        app = create_app(args.path, slow_ms=args.slow_ms,
                         missing_ms=args.missing_ms, limit=args.last)
        if not _nonnegative(args.interval) or args.interval <= 0:
            raise ValueError("interval must be positive and finite")
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be 1..65535")
    except ValueError as exc:
        parser.error(str(exc))
    collect = app.extensions["friday_snapshot"]
    if args.check:
        data = collect()
        print(json.dumps(data, indent=2))
        return 2 if data["source_error"] else int(bool(data["alerts"]))
    stop = threading.Event()
    monitor = threading.Thread(target=AlertMonitor(collect).run,
                               args=(stop, args.interval), daemon=True)
    monitor.start()
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        stop.set()
        monitor.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
