"""Offline business-task evidence recording and manual/assistant comparison."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

from friday.daemon import FRIDAY_HOME

DEFAULT_DB = FRIDAY_HOME / "business-comparison.sqlite3"
LIMITATIONS = [
    "These fixed synthetic tasks do not represent all business work.",
    "Quality uses exact fixture checks; it does not assess prose or real-world validity.",
    "Times and costs are operator-recorded evidence; this tool does not verify them.",
    "Latency must include reading, execution and checking the final answer for both methods.",
    "Model costs cover recorded API charges only; labour, hardware and voice costs are excluded.",
    "Missing measurements stay unknown. A failed or missing baseline provides no latency comparison.",
    "Observed latency differences do not establish productivity gains or causality.",
]


def fixtures() -> dict:
    return json.loads(files("friday").joinpath("data/business_tasks.json").read_text(encoding="utf-8"))


def _number(value, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def _text(value, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _equal(actual, expected) -> bool:
    # JSON booleans must not count as 0/1; ordinary numeric representations may match.
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_equal(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_equal(a, e) for a, e in zip(actual, expected, strict=True)))
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def evaluate(task: dict, observation: dict) -> dict:
    output = observation["output"]
    checks = {key: isinstance(output, dict) and key in output and _equal(output[key], value)
              for key, value in task["expected"].items()}
    checks["no_extra_fields"] = isinstance(output, dict) and not output.keys() - task["expected"].keys()
    passed = sum(checks.values())
    return {
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "quality_ratio": passed / len(checks),
        "complete": observation["status"] == "completed" and all(checks.values()),
    }


def validate(observation: dict, suite: dict) -> dict:
    if not isinstance(observation, dict):
        raise TypeError("observation must be a JSON object")
    if type(observation.get("suite_version")) is not int or observation["suite_version"] != suite["version"]:
        raise ValueError(f"suite_version must be {suite['version']}")
    task = next((t for t in suite["tasks"] if t["id"] == observation.get("task_id")), None)
    if task is None:
        raise ValueError("task_id must identify a fixed business task")
    for name in ("trial_id", "evidence"):
        _text(observation.get(name), name)
    if observation.get("method") not in ("manual", "assistant"):
        raise ValueError("method must be manual or assistant")
    if observation.get("status") not in ("completed", "failed"):
        raise ValueError("status must be completed or failed")
    _number(observation.get("elapsed_seconds"), "elapsed_seconds")
    if "output" not in observation or not (observation["output"] is None or isinstance(observation["output"], dict)):
        raise ValueError("output must be a JSON object, or null for a failed attempt")
    if observation["status"] == "completed" and observation["output"] is None:
        raise ValueError("completed attempts require an output object")
    limitations = observation.get("limitations")
    if not isinstance(limitations, list) or any(not isinstance(x, str) or not x.strip() for x in limitations):
        raise ValueError("limitations must be an array of nonempty strings (or [])")
    if observation["status"] == "failed" and not limitations:
        raise ValueError("failed attempts require a limitation explaining the failure")
    cost = observation.get("model_cost_usd")
    if cost is not None:
        _number(cost, "model_cost_usd")
        _text(observation.get("cost_evidence"), "cost_evidence")
    if observation["method"] == "assistant":
        _text(observation.get("model"), "model")
    elif cost is not None or observation.get("model") is not None:
        raise ValueError("manual attempts must leave model and model_cost_usd null")
    # Reject NaN/Infinity anywhere in the evidence, including nested output.
    json.dumps(observation, allow_nan=False)
    return task


def record(db: Path, observation: dict) -> dict:
    suite = fixtures()
    task = validate(observation, suite)
    result = {**observation, "model_cost_usd": observation.get("model_cost_usd"),
              "recorded_at": datetime.now(UTC).isoformat(),
              **evaluate(task, observation)}
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=2)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS business_task_runs (
            suite_version INTEGER NOT NULL, task_id TEXT NOT NULL,
            trial_id TEXT NOT NULL, method TEXT NOT NULL, result_json TEXT NOT NULL,
            PRIMARY KEY (suite_version, task_id, trial_id, method))""")
        try:
            conn.execute("INSERT INTO business_task_runs VALUES (?, ?, ?, ?, ?)",
                         (suite["version"], task["id"], observation["trial_id"], observation["method"],
                          json.dumps(result, allow_nan=False)))
        except sqlite3.IntegrityError as exc:
            raise ValueError("this task/trial/method already exists; use a new trial_id") from exc
    return result


def report(db: Path) -> dict:
    suite = fixtures()
    results = []
    if db.exists():
        # Reporting never creates or alters the evidence database.
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
            results = [json.loads(row[0]) for row in conn.execute(
                "SELECT result_json FROM business_task_runs WHERE suite_version = ? "
                "ORDER BY task_id, trial_id, method", (suite["version"],))]
    tasks = []
    for task in suite["tasks"]:
        trials = {}
        for result in results:
            if result["task_id"] == task["id"]:
                trial = trials.setdefault(result["trial_id"], {"manual": None, "assistant": None})
                trial[result["method"]] = result
        for trial in trials.values():
            manual, assistant = trial["manual"], trial["assistant"]
            comparable = manual and assistant and manual["complete"] and assistant["complete"]
            trial["assistant_minus_manual_seconds"] = (
                assistant["elapsed_seconds"] - manual["elapsed_seconds"] if comparable else None)
        tasks.append({"task_id": task["id"], "title": task["title"], "trials": trials})
    return {"suite_version": suite["version"], "recorded_attempts": len(results),
            "limitations": LIMITATIONS, "tasks": tasks}


def format_report(data: dict) -> str:
    lines = [f"Business-task comparison | suite {data['suite_version']} | {data['recorded_attempts']} recorded attempts"]
    for task in data["tasks"]:
        lines.append(f"\n{task['task_id']}: {task['title']}")
        if not task["trials"]:
            lines.append("  No measurements recorded.")
        for trial_id, trial in task["trials"].items():
            lines.append(f"  Trial: {trial_id}")
            for method in ("manual", "assistant"):
                result = trial[method]
                if result is None:
                    lines.append(f"    {method}: missing baseline" if method == "manual" else "    assistant: missing attempt")
                    continue
                cost = result["model_cost_usd"]
                cost_label = ("n/a (manual)" if method == "manual" else
                              "unknown" if cost is None else f"USD {cost}")
                lines.append(
                    f"    {method}: {result['status']}; quality {result['checks_passed']}/{result['checks_total']} checks; "
                    f"complete={result['complete']}; latency {result['elapsed_seconds']:g}s; model cost {cost_label}")
                if method == "assistant":
                    lines.append(f"      Model: {result['model']}")
                lines.append(f"      Evidence: {result['evidence']}")
                if cost is not None:
                    lines.append(f"      Cost evidence: {result['cost_evidence']}")
                failed = [key for key, passed in result["checks"].items() if not passed]
                if failed:
                    lines.append(f"      Failed checks: {', '.join(failed)}")
                lines.extend(f"      Limitation: {value}" for value in result["limitations"])
            delta = trial["assistant_minus_manual_seconds"]
            lines.append("    Assistant minus manual latency: " +
                         ("unavailable (requires both attempts to pass all checks)." if delta is None else f"{delta:+g}s (observed)."))
    lines.append("\nLimitations:")
    lines.extend(f"- {value}" for value in data["limitations"])
    return "\n".join(lines)


def command(args: argparse.Namespace) -> int:
    try:
        if args.comparison_action == "tasks":
            suite = fixtures()
            public = {"suite_version": suite["version"], "tasks": [
                {k: v for k, v in task.items() if k != "expected"} for task in suite["tasks"]]}
            print(json.dumps(public, indent=2))
        elif args.comparison_action == "record":
            observation = json.loads(Path(args.observation).read_text(encoding="utf-8"))
            print(json.dumps(record(Path(args.db), observation), indent=2))
        else:
            data = report(Path(args.db))
            print(json.dumps(data, indent=2) if args.json else format_report(data))
        return 0
    except (OSError, TypeError, ValueError, OverflowError, sqlite3.Error) as exc:
        print(f"Business-task comparison: {exc}", file=sys.stderr)
        return 1


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="record and compare fixed business-task executions")
    actions = parser.add_subparsers(dest="comparison_action", required=True)
    tasks = actions.add_parser("tasks", help="show fixed prompts and quality criteria as JSON")
    tasks.set_defaults(func=command)
    record_parser = actions.add_parser("record", help="record one measured execution from JSON")
    record_parser.add_argument("observation", help="JSON evidence file; see docs/business-comparison.md")
    record_parser.add_argument("--db", default=str(DEFAULT_DB))
    record_parser.set_defaults(func=command)
    report_parser = actions.add_parser("report", help="show all recorded trials and limitations")
    report_parser.add_argument("--db", default=str(DEFAULT_DB))
    report_parser.add_argument("--json", action="store_true")
    report_parser.set_defaults(func=command)
