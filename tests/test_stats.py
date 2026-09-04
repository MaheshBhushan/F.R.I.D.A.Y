"""`friday stats`: percentiles from spans.jsonl, filters, and JSON output."""

from __future__ import annotations

import json

from friday import cli
from friday.scripts import percentiles

RECORDS = [
    {"turn_kind": "reasoning", "stages": {"speech_ended_vad": 100_000_000, "first_token": 1_100_000_000, "llm_sent": 200_000_000, "task_complete": 3_000_000_000}},
    {"turn_kind": "reasoning", "stages": {"speech_ended_vad": 100_000_000, "first_token": 2_100_000_000, "llm_sent": 200_000_000, "task_complete": 4_000_000_000}},
    {"turn_kind": "state_query", "stages": {"speech_ended_vad": 100_000_000, "task_complete": 140_000_000}},
    {"turn_kind": "reflex", "stages": {"ack_audible": 50_000_000}},
]


def _write(tmp_path):
    path = tmp_path / "spans.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return path


def test_summarize_reports_percentiles_and_deltas():
    summary = percentiles.summarize(RECORDS)
    assert summary["turns"] == 4
    assert summary["kinds"] == {"reasoning": 2, "state_query": 1, "reflex": 1}
    ttft = next(d for d in summary["deltas"] if d["name"] == "first_token - llm_sent")
    assert ttft["count"] == 2 and ttft["p50_ms"] == 1400.0
    e2e = next(d for d in summary["deltas"] if d["name"] == "task_complete - speech_ended_vad")
    assert e2e["count"] == 3
    empty = next(d for d in summary["deltas"] if d["name"] == "ack_audible - speech_ended_vad")
    assert empty["count"] == 0 and empty["p50_ms"] is None


def test_filters_by_kind_and_last():
    assert len(percentiles.filter_records(RECORDS, kind="reasoning")) == 2
    assert percentiles.filter_records(RECORDS, last=1)[0]["turn_kind"] == "reflex"


def test_cli_stats_table_and_json(tmp_path, capsys):
    path = _write(tmp_path)
    assert cli.main(["stats", "--path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "4 turns" in out and "first_token - llm_sent" in out
    assert "mean" not in out.lower()

    assert cli.main(["stats", "--path", str(path), "--kind", "state_query", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["turns"] == 1 and data["kinds"] == {"state_query": 1}


def test_cli_stats_empty_is_not_an_error(tmp_path, capsys):
    assert cli.main(["stats", "--path", str(tmp_path / "none.jsonl"), "--kind", "reflex"]) == 0
    assert "no turns recorded" in capsys.readouterr().out
