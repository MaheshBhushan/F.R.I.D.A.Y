"""Replay local traces and advance deterministic clocks; no voice services needed."""

import json
import threading
from pathlib import Path

import pytest

from friday import dashboard
from friday.core import spans

FIXTURE = Path(__file__).parent / "fixtures" / "runtime_traces.jsonl"


@pytest.fixture
def delayed_response(tmp_path, monkeypatch):
    clock = [0]
    monkeypatch.setattr(spans.time, "time", lambda: 100 + clock[0] / 1e9)
    monkeypatch.setattr(spans.time, "perf_counter_ns", lambda: clock[0])
    turn = spans.TurnSpan("reasoning", path=tmp_path / "spans.jsonl")
    turn.mark("speech_ended_vad")
    turn.mark("intent_classified")
    return turn, clock


def test_replay_dashboard_exposes_real_percentiles_and_alerts():
    app = dashboard.create_app(FIXTURE)
    client = app.test_client()
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json
    assert data["summary"]["turns"] == 4
    assert {a["turn_id"]: a["code"] for a in data["alerts"]} == {
        "delayed": "slow_response", "missing": "missing_response"}
    delta = next(row for row in data["summary"]["deltas"]
                 if row["name"] == "first_content_audio - speech_ended_vad")
    assert delta["count"] == 2
    assert delta["p50_ms"] == 2100
    page = client.get("/")
    assert page.status_code == 200
    assert b"slow_response" in page.data and b"missing_response" in page.data
    assert b"llm_wait" in page.data and b"2100" in page.data
    assert page.headers["Cache-Control"] == "no-store"
    assert client.post("/api/metrics", json={}).status_code == 405


def test_delayed_live_response_alerts_without_browser_then_recovers(delayed_response, monkeypatch):
    turn, clock = delayed_response
    collect = dashboard.create_app(turn.path).extensions["friday_snapshot"]
    emitted = []
    monkeypatch.setattr(dashboard.events, "emit", lambda kind, **fields: emitted.append((kind, fields)))
    monitor = dashboard.AlertMonitor(collect)
    clock[0] = 9_999_000_000
    monitor.poll()
    assert emitted == []
    clock[0] = 10_000_000_000
    monitor.poll()
    monitor.poll()
    assert [e[1]["code"] for e in emitted] == ["missing_response"]
    turn.mark("first_content_audio")
    monitor.poll()
    assert [(kind, fields["code"]) for kind, fields in emitted] == [
        ("alert", "missing_response"), ("alert", "slow_response"),
        ("alert-cleared", "missing_response")]
    turn.mark("task_complete")
    turn.write()
    assert list(spans.active_spans_dir(turn.path).glob("*.json")) == []
    assert collect()["summary"]["turns"] == 1
    assert collect()["alerts"][0]["elapsed_ms"] == 10000
    monitor.poll()
    assert len(emitted) == 3


@pytest.mark.parametrize("delay_ns,expected", [(2_999_000_000, []), (3_000_000_000, []),
                                                (3_001_000_000, ["slow_response"])])
def test_slow_threshold_and_final_trace_replay(delayed_response, delay_ns, expected):
    turn, clock = delayed_response
    clock[0] = delay_ns
    turn.mark("first_content_audio")
    turn.mark("task_complete")
    turn.write()
    data = dashboard.snapshot(turn.path, slow_ms=3000, missing_ms=10000, limit=200, now=200)
    assert [a["code"] for a in data["alerts"]] == expected


def test_reasoning_ack_is_not_a_response_but_reflex_and_text_completion_are():
    records, _ = dashboard.read_traces(FIXTURE, 200)
    missing = next(r for r in records if r["turn_id"] == "missing")
    reflex = {**missing, "turn_id": "reflex", "turn_kind": "reflex"}
    text = {**missing, "turn_id": "text", "stages": {**missing["stages"], "task_complete": 500000000}}
    assert dashboard.evaluate_alerts([reflex, text], now=1700000020,
                                     slow_ms=3000, missing_ms=10000) == []
    assert dashboard.evaluate_alerts([missing], now=1700000012.1,
                                     slow_ms=3000, missing_ms=10000)[0]["code"] == "missing_response"


def test_malformed_partial_and_empty_logs(tmp_path):
    path = tmp_path / "spans.jsonl"
    client = dashboard.create_app(path).test_client()
    assert client.get("/api/metrics").json["summary"]["turns"] == 0
    assert b"No turns recorded" in client.get("/").data
    valid = FIXTURE.read_text().splitlines()[0]
    path.write_text(valid + '\n[]\n{"stages":null}\n{"unfinished":\n')
    data = client.get("/api/metrics").json
    assert data["summary"]["turns"] == 1
    assert data["invalid_records"] == 3
    # A writer can finish a partial line before the next poll.
    path.write_text(valid + "\n")
    assert client.get("/api/metrics").json["invalid_records"] == 0


def test_window_final_precedence_and_html_escaping(tmp_path):
    path = tmp_path / "spans.jsonl"
    records, _ = dashboard.read_traces(FIXTURE, 200)
    record = records[0]
    record["turn_id"] = "<script>alert(1)</script>"
    active = spans.active_spans_dir(path)
    active.mkdir()
    (active / "pending.json").write_text(json.dumps({**record, "stages": {"intent_classified": 0}}))
    path.write_text(json.dumps(record) + "\n" + json.dumps(records[1]) + "\n")
    client = dashboard.create_app(path).test_client()
    assert client.get("/api/metrics").json["summary"]["turns"] == 2
    assert b"<script>" not in client.get("/").data
    assert b"&lt;script&gt;" in client.get("/").data
    assert dashboard.create_app(path, limit=1).test_client().get("/api/metrics").json["summary"]["turns"] == 1


def test_source_errors_are_visible_and_monitor_runs_without_requests(tmp_path, monkeypatch):
    client = dashboard.create_app(tmp_path).test_client()  # A directory cannot be read as JSONL.
    assert client.get("/api/metrics").status_code == 503
    assert b"Trace read failed" in client.get("/").data
    stop = threading.Event()
    emitted = []
    monkeypatch.setattr(dashboard.events, "emit", lambda kind, **fields: emitted.append(kind))

    def collect():
        stop.set()
        return client.get("/api/metrics").json

    dashboard.AlertMonitor(collect).run(stop, .01)
    assert emitted == ["alert"]


def test_check_command_and_invalid_config(tmp_path, capsys):
    assert dashboard.main(["--path", str(tmp_path / "none"), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["turns"] == 0
    assert dashboard.main(["--path", str(FIXTURE), "--check"]) == 1
    assert len(json.loads(capsys.readouterr().out)["alerts"]) == 2
    assert dashboard.main(["--path", str(tmp_path), "--check"]) == 2
    for flag, value in [("--slow-ms", "nan"), ("--missing-ms", "-1"),
                        ("--last", "0"), ("--interval", "inf"), ("--port", "0")]:
        with pytest.raises(SystemExit) as exc:
            dashboard.main([flag, value, "--check"])
        assert exc.value.code == 2


def test_checkpoint_failure_does_not_break_voice_turn(delayed_response, monkeypatch):
    turn, clock = delayed_response

    def fail(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(Path, "write_text", fail)
    clock[0] = 500000000
    assert turn.mark("first_content_audio") == 500000000
    assert turn.stages["first_content_audio"] == 500000000


def test_checkpoint_cleanup_failure_preserves_final_trace(delayed_response, monkeypatch):
    turn, clock = delayed_response
    clock[0] = 400000000
    turn.mark("first_content_audio")

    def fail(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(Path, "unlink", fail)
    turn.write()
    records, invalid = dashboard.read_traces(turn.path, 200)
    assert invalid == 0 and len(records) == 1
    assert records[0]["stages"]["first_content_audio"] == 400000000
    # Late audio callbacks cannot resurrect checkpoints after the final write.
    turn.mark("tts_started")
    assert json.loads(next(spans.active_spans_dir(turn.path).glob("*.json")).read_text())["stages"].get("tts_started") is None


def test_monitor_preserves_alerts_during_read_failure(monkeypatch):
    state = {"alerts": [{"turn_id": "missing", "code": "missing_response"}], "source_error": None}
    emitted = []
    monkeypatch.setattr(dashboard.events, "emit", lambda kind, **fields: emitted.append(kind))
    monitor = dashboard.AlertMonitor(lambda: state)
    monitor.poll()
    state.update(alerts=[], source_error="read failed")
    monitor.poll()
    assert emitted == ["alert", "alert"]
    assert ("missing", "missing_response") in monitor.active
    state["source_error"] = None
    monitor.poll()
    assert emitted == ["alert", "alert", "alert-cleared", "alert-cleared"]
