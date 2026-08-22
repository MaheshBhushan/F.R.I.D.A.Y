"""The event log is the only window into a live voice turn, so its own
failure modes matter: a log that swallows lines, or that lets a reply forge
extra entries, is worse than no log."""

import io

from friday.core import events


def _capture(monkeypatch, level="info"):
    monkeypatch.setenv("FRIDAY_LOG", level)
    buf = io.StringIO()
    events.set_stream(buf)
    return buf


def test_info_events_are_written(monkeypatch):
    buf = _capture(monkeypatch)
    events.emit("wake", model="alexa", score=0.812)
    events.set_stream(None)
    line = buf.getvalue().strip()
    assert "wake" in line and "model=alexa" in line and "score=0.812" in line


def test_debug_is_suppressed_at_info(monkeypatch):
    buf = _capture(monkeypatch, "info")
    events.debug("wake-score", score=0.3)
    events.set_stream(None)
    assert buf.getvalue() == ""


def test_debug_appears_at_debug(monkeypatch):
    buf = _capture(monkeypatch, "debug")
    events.debug("wake-score", score=0.3)
    events.set_stream(None)
    assert "wake-score" in buf.getvalue()


def test_silent_suppresses_even_info(monkeypatch):
    buf = _capture(monkeypatch, "silent")
    events.emit("wake", model="alexa")
    events.set_stream(None)
    assert buf.getvalue() == ""


def test_none_fields_are_dropped_not_printed(monkeypatch):
    buf = _capture(monkeypatch)
    events.emit("route", tier="reflex", why=None)
    events.set_stream(None)
    assert "why" not in buf.getvalue()


def test_a_multiline_reply_cannot_forge_log_entries(monkeypatch):
    # A reply is model output. If it were printed raw, "\n22:00:00 wake ..."
    # inside it would read as a genuine event to anyone tailing the log.
    buf = _capture(monkeypatch)
    events.emit("reply", events.quote("line one\nline two"))
    events.set_stream(None)
    assert buf.getvalue().count("\n") == 1
    assert "line one line two" in buf.getvalue()


def test_quote_truncates_long_text():
    out = events.quote("x" * 500)
    assert len(out) < 200 and out.endswith('…"')


def test_quote_renders_empty_as_empty_quotes():
    # An empty transcript is the signature of "she woke but heard nothing",
    # so it has to be visible rather than a blank gap in the line.
    assert events.quote("") == '""'
    assert events.quote(None) == '""'


def test_a_broken_stream_does_not_raise(monkeypatch):
    class Exploding:
        def write(self, *_):
            raise OSError("log device went away")
        def flush(self):
            pass

    monkeypatch.setenv("FRIDAY_LOG", "info")
    events.set_stream(Exploding())
    try:
        events.emit("wake", model="alexa")  # must not propagate
    finally:
        events.set_stream(None)


# --------------------------------------------------- bounded shutdown

def test_shutdown_drain_is_under_the_units_stop_timeout():
    """A drain longer than TimeoutStopSec means systemd resolves the hang with
    SIGKILL, which skips echo-cancel cleanup and leaks the module. The budget
    only buys anything if it expires first."""
    import re
    from pathlib import Path

    from friday.core.app import SHUTDOWN_DRAIN_S

    unit = (Path(__file__).resolve().parent.parent
            / "deploy" / "friday.service").read_text()
    match = re.search(r"^TimeoutStopSec=(\d+)", unit, re.M)
    assert match, "unit has no TimeoutStopSec; the drain budget is unanchored"
    assert SHUTDOWN_DRAIN_S < int(match.group(1))


def test_unit_reaps_echo_cancel_after_stop():
    """The in-process cleanup cannot run if the daemon was SIGKILLed, so the
    unit needs its own backstop -- a leaked module renumbers every other
    application's capture device list."""
    from pathlib import Path

    unit = (Path(__file__).resolve().parent.parent
            / "deploy" / "friday.service").read_text()
    assert "ExecStopPost=-" in unit, "a failed sweep must not fail `systemctl stop`"
    assert "reap" in unit
