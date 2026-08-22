"""Tests for the `friday` command and its lifecycle layer.

Nothing here starts a real daemon: the process-management logic is what needs
proving, and it is exactly the part that must not be exercised by spawning
audio-holding processes inside a test run. The subprocess and signal seams are
injected instead.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from friday import cli, daemon


# ------------------------------------------------------------ pid handling

def test_stale_pid_file_is_cleaned_and_not_signalled(tmp_path, monkeypatch):
    """A recycled PID must never be signalled.

    A pidfile surviving a reboot can name an unrelated process, and `kill`
    does not ask whether you meant it.
    """
    pid_file = tmp_path / "friday.pid"
    # Our own PID: alive and signalable, so the guard cannot short-circuit on
    # a dead process -- it has to reject it on the command line, which is
    # pytest, not FRIDAY. (PID 1 would raise PermissionError instead, which is
    # a different branch: alive but not ours to manage, so not ours to clear.)
    pid_file.write_text(f"{os.getpid()}\n")
    monkeypatch.setattr(daemon, "PID_PATH", pid_file)
    assert daemon.read_pid() is None
    assert not pid_file.exists(), "stale pidfile should be removed"


def test_a_live_but_unowned_pid_is_left_alone(tmp_path, monkeypatch):
    # PID 1 is alive and not ours: kill(1, 0) raises PermissionError. Reporting
    # "not running" is right, but deleting the file would be presumptuous --
    # and clearing it is the step before signalling something.
    pid_file = tmp_path / "friday.pid"
    pid_file.write_text("1\n")
    monkeypatch.setattr(daemon, "PID_PATH", pid_file)
    assert daemon.read_pid() is None
    assert pid_file.exists()


def test_dead_pid_is_cleaned_up(tmp_path, monkeypatch):
    pid_file = tmp_path / "friday.pid"
    monkeypatch.setattr(daemon, "PID_PATH", pid_file)
    # Reap a real child so the PID is certainly dead and certainly not reused.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    pid_file.write_text(f"{proc.pid}\n")
    assert daemon.read_pid() is None
    assert not pid_file.exists()


def test_garbage_pid_file_does_not_raise(tmp_path, monkeypatch):
    pid_file = tmp_path / "friday.pid"
    monkeypatch.setattr(daemon, "PID_PATH", pid_file)
    for junk in ("", "   ", "not-a-pid", "-5", "0"):
        pid_file.write_text(junk)
        assert daemon.read_pid() is None


def test_is_friday_rejects_an_unrelated_command_line(monkeypatch):
    monkeypatch.setattr(daemon, "_cmdline", lambda pid: "/usr/bin/sleep 100")
    assert not daemon._is_friday(1234)
    monkeypatch.setattr(daemon, "_cmdline",
                        lambda pid: "/x/.venv/bin/python -m friday")
    assert daemon._is_friday(1234)


def test_is_friday_rejects_a_merely_similar_path(monkeypatch):
    # A checkout at ~/friday running something else entirely must not match.
    monkeypatch.setattr(daemon, "_cmdline",
                        lambda pid: "vim /home/me/friday/notes.md")
    assert not daemon._is_friday(1234)


# -------------------------------------------------------------- supervisor

def test_supervisor_delegates_when_the_unit_exists(monkeypatch):
    """An installed unit means systemd owns the lifecycle.

    Signalling the process directly while systemd has Restart=always would see
    it resurrected within seconds, making `friday stop` look broken.
    """
    monkeypatch.setattr(daemon, "unit_installed", lambda: True)
    assert daemon.supervisor() is daemon.Supervisor.SYSTEMD
    monkeypatch.setattr(daemon, "unit_installed", lambda: False)
    assert daemon.supervisor() is daemon.Supervisor.DIRECT


def test_stop_delegates_to_systemctl_and_sends_no_signals(monkeypatch):
    calls: list[tuple] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(daemon, "unit_installed", lambda: True)
    monkeypatch.setattr(daemon, "_systemctl",
                        lambda *a, **k: (calls.append(a), Result())[1])
    monkeypatch.setattr(daemon, "read_pid",
                        lambda: pytest.fail("must not read the pidfile under systemd"))

    def _no_kill(*a, **k):
        pytest.fail("must not signal the process directly under systemd")

    monkeypatch.setattr(os, "kill", _no_kill)
    ok, note = daemon.stop()
    assert ok and "systemd" in note
    assert ("stop", daemon.UNIT_NAME) == calls[0]


# -------------------------------------------------------------------- stop

def test_stop_escalates_and_reaps_on_a_signal_deaf_daemon(monkeypatch):
    """SIGINT, then SIGTERM, then SIGKILL -- and clean up after a hard kill.

    A daemon killed before it could unwind never released its echo-cancel
    module, and a leaked virtual source renumbers every other application's
    capture device list. That surfaces hours later in an unrelated app with no
    visible link to FRIDAY, so the crash path repairs it here.
    """
    sent: list[int] = []
    reaped: list[int] = []

    monkeypatch.setattr(daemon, "unit_installed", lambda: False)
    monkeypatch.setattr(daemon, "read_pid", lambda: 4242)
    monkeypatch.setattr(daemon, "_clear_pid", lambda: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    # Deaf to everything except SIGKILL.
    monkeypatch.setattr(daemon, "_wait_gone",
                        lambda pid, timeout: sent[-1] == signal.SIGKILL)
    monkeypatch.setattr(daemon, "reap_echo_cancel",
                        lambda: (reaped.append(1), 1)[1])

    ok, note = daemon.stop()
    assert ok
    assert sent == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert reaped, "a hard kill must trigger echo-cancel cleanup"
    assert "cleaned up 1 leaked echo-cancel module" in note


def test_graceful_stop_does_not_reap_anything(monkeypatch):
    # The daemon unwound on its own, so it already released the module. Reaping
    # here could unload one a *different* process had just loaded.
    monkeypatch.setattr(daemon, "unit_installed", lambda: False)
    monkeypatch.setattr(daemon, "read_pid", lambda: 4242)
    monkeypatch.setattr(daemon, "_clear_pid", lambda: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(daemon, "_wait_gone", lambda pid, timeout: True)
    monkeypatch.setattr(daemon, "reap_echo_cancel",
                        lambda: pytest.fail("must not reap after a clean stop"))
    ok, note = daemon.stop()
    assert ok and note == "stopped"


def test_stop_when_not_running_is_success_not_an_error(monkeypatch):
    # `friday stop` in a script must be idempotent.
    monkeypatch.setattr(daemon, "unit_installed", lambda: False)
    monkeypatch.setattr(daemon, "read_pid", lambda: None)
    monkeypatch.setattr(daemon, "_clear_pid", lambda: None)
    ok, note = daemon.stop()
    assert ok and note == "not running"


# ------------------------------------------------------------------- start

def test_start_is_idempotent_and_spawns_nothing_when_up(monkeypatch):
    monkeypatch.setattr(daemon, "status",
                        lambda: daemon.Status(True, True, 7, daemon.Supervisor.DIRECT))
    monkeypatch.setattr(daemon, "_spawn",
                        lambda: pytest.fail("must not spawn a second daemon"))
    result = daemon.start()
    assert result.running and result.detail == "already running"


def test_start_bails_early_when_the_process_dies_during_boot(monkeypatch):
    """Don't burn the full 45s timeout on a process that already exited."""
    monkeypatch.setattr(daemon, "unit_installed", lambda: False)
    monkeypatch.setattr(daemon, "_spawn", lambda: 999)
    monkeypatch.setattr(daemon, "read_pid", lambda: None)
    monkeypatch.setattr(daemon, "health", lambda timeout=0: None)
    result = daemon.start(timeout=30.0)
    assert not result.running
    assert "exited during startup" in result.detail


# ------------------------------------------------------------------- parser

def test_every_subcommand_is_wired_to_a_handler():
    parser = cli.build_parser()
    for name in ("start", "stop", "restart", "status", "logs", "ask", "say",
                 "smoke", "doctor", "install", "uninstall", "token"):
        args = parser.parse_args([name] + (["x"] if name in {"ask", "say"} else []))
        assert callable(getattr(args, "func", None)), f"{name} has no handler"


def test_ask_joins_its_words_so_quoting_is_optional():
    args = cli.build_parser().parse_args(["ask", "what", "branch", "am", "i", "on"])
    assert " ".join(args.text) == "what branch am i on"


def test_ask_and_say_refuse_cleanly_when_stopped(monkeypatch, capsys):
    # Exit 2 distinguishes "not running" from "ran and failed" (1), so a script
    # can tell the difference without parsing text.
    monkeypatch.setattr(daemon, "status",
                        lambda: daemon.Status(False, False, None,
                                              daemon.Supervisor.DIRECT))
    for name in ("ask", "say"):
        args = cli.build_parser().parse_args([name, "hello"])
        assert args.func(args) == 2
    assert "not running" in capsys.readouterr().err
