"""Business-task comparisons use synthetic test evidence and no external services."""

import copy
import json
import sqlite3

import pytest

from friday import cli, comparison


def observation(method="assistant", **changes):
    return {
        "suite_version": 1, "task_id": "invoice-check", "trial_id": "test-1",
        "method": method, "status": "completed", "elapsed_seconds": 15,
        "output": {"net_eur": 220, "vat_eur": 41.8, "gross_eur": 261.8,
                   "overcharge_eur": 8.2, "decision": "hold"},
        "model": "test-model" if method == "assistant" else None,
        "model_cost_usd": None, "evidence": "Synthetic unit-test observation, not a measurement.",
        "limitations": [], **changes,
    }


def test_cli_records_persists_and_reports_paired_observations(tmp_path, capsys):
    db = tmp_path / "evidence.sqlite3"
    for attempt in (observation("manual", elapsed_seconds=60),
                    observation(model_cost_usd=0.004, cost_evidence="Synthetic API charge for this unit test.")):
        source = tmp_path / "attempt.json"
        source.write_text(json.dumps(attempt))
        assert cli.main(["compare", "record", str(source), "--db", str(db)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["complete"] is True
        assert result["checks_passed"] == result["checks_total"] == 6
        assert result["recorded_at"]

    assert cli.main(["compare", "report", "--db", str(db), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recorded_attempts"] == 2
    trial = report["tasks"][0]["trials"]["test-1"]
    assert trial["assistant_minus_manual_seconds"] == -45
    assert trial["assistant"]["model_cost_usd"] == 0.004
    assert trial["manual"]["model_cost_usd"] is None
    assert trial["assistant"]["output"]["gross_eur"] == 261.8
    assert cli.main(["compare", "report", "--db", str(db)]) == 0
    text = capsys.readouterr().out
    for expected in ("quality 6/6", "latency 15s", "USD 0.004", "n/a (manual)",
                     "-45s (observed)", "test-model", "productivity gains", "synthetic"):
        assert expected in text


def test_missing_results_remain_unknown_and_report_does_not_create_db(tmp_path, capsys):
    db = tmp_path / "missing.sqlite3"
    assert cli.main(["compare", "report", "--db", str(db), "--json"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["recorded_attempts"] == 0 and not db.exists()
    assert len(empty["tasks"]) == 3
    assert all(task["trials"] == {} for task in empty["tasks"])
    comparison.record(db, observation())
    data = comparison.report(db)
    trial = data["tasks"][0]["trials"]["test-1"]
    assert trial["manual"] is None
    assert trial["assistant_minus_manual_seconds"] is None
    assert trial["assistant"]["model_cost_usd"] is None
    text = comparison.format_report(data)
    assert "missing baseline" in text and "model cost unknown" in text


def test_no_cross_trial_pairing_and_no_overwrite(tmp_path):
    db = tmp_path / "runs.sqlite3"
    original = observation()
    comparison.record(db, original)
    comparison.record(db, observation("manual", trial_id="other-trial"))
    before = db.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        comparison.record(db, observation(elapsed_seconds=1))
    assert db.read_bytes() == before
    data = comparison.report(db)
    assert data["recorded_attempts"] == 2
    assert all(t["assistant_minus_manual_seconds"] is None for t in data["tasks"][0]["trials"].values())
    assert original == observation(), "Recording must not mutate the caller's evidence"


def test_partial_quality_and_failures_prevent_latency_claims(tmp_path):
    db = tmp_path / "runs.sqlite3"
    wrong = observation("manual")
    wrong["output"]["decision"] = "approve"
    comparison.record(db, wrong)
    result = comparison.record(db, observation())
    assert result["complete"] is True
    trial = comparison.report(db)["tasks"][0]["trials"]["test-1"]
    assert trial["manual"]["checks_passed"] == 5
    assert trial["manual"]["complete"] is False
    assert trial["assistant_minus_manual_seconds"] is None
    failed = comparison.record(db, observation(trial_id="failed", status="failed", output=None,
                                              limitations=["Assistant returned invalid JSON."]))
    assert failed["quality_ratio"] == 0 and failed["complete"] is False
    assert "Assistant returned invalid JSON" in comparison.format_report(comparison.report(db))


def test_fixture_rubrics_detect_missing_dates_extra_claims_and_boolean_amounts():
    suite = comparison.fixtures()
    assert {t["id"] for t in suite["tasks"]} == {"invoice-check", "meeting-actions", "ticket-triage"}
    for task in suite["tasks"]:
        attempt = observation(task_id=task["id"], output=copy.deepcopy(task["expected"]))
        assert comparison.evaluate(task, attempt)["complete"] is True
        attempt["output"]["invented_claim"] = "approved by manager"
        assert comparison.evaluate(task, attempt)["complete"] is False
    meeting = suite["tasks"][1]
    attempt = observation(output=copy.deepcopy(meeting["expected"]))
    attempt["output"]["actions"][1]["due_date"] = "2026-09-21"
    assert not comparison.evaluate(meeting, attempt)["checks"]["actions"]
    assert not comparison._equal(True, 1)
    assert comparison._equal(220.0, 220)
    assert not comparison._equal([1], [1, 2])


@pytest.mark.parametrize("changes", [
    {"suite_version": 2}, {"suite_version": True}, {"task_id": "unknown"},
    {"method": "simulated"}, {"trial_id": " "}, {"evidence": ""},
    {"elapsed_seconds": -1}, {"elapsed_seconds": True}, {"elapsed_seconds": float("inf")},
    {"elapsed_seconds": float("nan")}, {"status": "success"}, {"output": "not JSON"},
    {"output": None}, {"limitations": "none"}, {"limitations": [""]},
    {"status": "failed", "output": None}, {"model": ""},
    {"model_cost_usd": -0.01, "cost_evidence": "test"},
    {"model_cost_usd": 0.01}, {"model_cost_usd": True, "cost_evidence": "test"},
    {"method": "manual", "model_cost_usd": 0.01, "cost_evidence": "test"},
    {"method": "manual", "model": "test-model"}, {"output": {"net_eur": float("nan")}},
])
def test_invalid_evidence_never_creates_a_database(tmp_path, changes):
    db = tmp_path / "runs.sqlite3"
    with pytest.raises(ValueError):
        comparison.record(db, observation(**changes))
    assert not db.exists()


def test_zero_known_cost_stays_distinct_from_unknown(tmp_path):
    db = tmp_path / "runs.sqlite3"
    result = comparison.record(db, observation(model_cost_usd=0, cost_evidence="Synthetic free-tier charge."))
    assert result["model_cost_usd"] == 0
    assert "USD 0" in comparison.format_report(comparison.report(db))


def test_public_prompts_exclude_answers_and_cli_errors_are_clear(tmp_path, capsys):
    assert cli.main(["compare", "tasks"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["suite_version"] == 1
    assert all(t["prompt"] and t["criteria"] and "expected" not in t for t in data["tasks"])
    source = tmp_path / "broken.json"
    db = tmp_path / "runs.sqlite3"
    for content in ("{broken", "[]", "null", '{}'):
        source.write_text(content)
        assert cli.main(["compare", "record", str(source), "--db", str(db)]) == 1
        assert "Business-task comparison:" in capsys.readouterr().err
    assert not db.exists()
    assert cli.main(["compare", "record", str(tmp_path / "missing"), "--db", str(db)]) == 1
    assert "No such file" in capsys.readouterr().err
    db.write_bytes(b"not a SQLite database")
    assert cli.main(["compare", "report", "--db", str(db)]) == 1
    assert "database" in capsys.readouterr().err


def test_report_filters_out_other_suite_versions(tmp_path):
    db = tmp_path / "runs.sqlite3"
    comparison.record(db, observation())
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE business_task_runs SET suite_version = 0")
    assert comparison.report(db)["recorded_attempts"] == 0
