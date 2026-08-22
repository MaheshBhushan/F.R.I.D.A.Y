"""Tests for T9 coding-agent delegation over tmux.

Every session created here lives on the dedicated `friday` tmux socket and
carries `agents.SESSION_PREFIX`, so a stray test can never see, write to, or
kill a session on the user's real tmux server. `tmux kill-server` is never
called; sessions are cleaned up individually with `agents.stop`.
"""

from __future__ import annotations

import asyncio
import functools
import shutil
import subprocess
import time

import pytest

from friday import agents
from friday.permissions import PermissionDenied, Risk, risk_of
from friday.tools import TOOLS, execute

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _real_tmux(args: list[str]) -> subprocess.CompletedProcess:
    """Talk to the user's *default* tmux server -- never the friday socket --
    to plant/inspect a foreign session for the isolation tests."""
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


@pytest.fixture(autouse=True)
def cleanup_friday_sessions():
    yield
    for session in agents.list_sessions():
        try:
            agents.stop(session)
        except Exception:
            pass
    # Killing the last session tears the friday tmux server down in the
    # background; give it a moment so the next test's `new-session` doesn't
    # race a server that is mid-exit (tmux -L friday is otherwise untouched
    # test-infra flakiness, not a correctness issue in agents.py).
    for _ in range(20):
        if agents._tmux(["list-sessions"]).returncode != 0:
            break
        time.sleep(0.05)


# --- 3. state-detection correctness (ported parser.rs marker lists) --------


@pytest.mark.parametrize(
    "text,expected",
    [
        # From parser.rs's own test suite (spinner_means_running).
        ("✻ Thinking… (12s · esc to interrupt)", "running"),
        # From parser.rs's own test suite (interrupt_hint_means_running).
        ("Bashing… (34s · ctrl+c to interrupt)", "running"),
        # From parser.rs's own test suite (prompt_box_is_idle).
        ("● Done! All tests pass.\n\n╭──────────╮\n│ ❯        │\n╰──────────╯\n", "idle"),
        # From parser.rs's own test suite (claude_permission_dialog_is_permission).
        (
            "Do you want to make this edit to main.rs?\n"
            " ❯ 1. Yes\n"
            "   2. Yes, allow all edits during this session\n"
            "   3. No, and tell Claude what to do differently\n",
            "permission",
        ),
        # From parser.rs's own test suite (yn_prompt_is_permission).
        ("Allow edits to config.py? (Y)es/(N)o [Yes]:", "permission"),
        # From parser.rs's own test suite (error_output_is_idle).
        ("error: failed to compile\n> ", "idle"),
    ],
)
def test_classify_status_matches_rust_parser(text, expected):
    assert agents.classify_status(text) == expected


def test_classify_status_old_spinner_scrolled_away_is_idle():
    # From parser.rs's own test suite (old_spinner_scrolled_away_is_idle):
    # a marker line far above the tail must not count.
    text = "✻ Thinking…\n" + "output line\n" * 10
    assert agents.classify_status(text) == "idle"


# --- 1. end-to-end: spawn, write task, detect completion via diffing -------


def test_spawn_write_task_and_detect_completion():
    # The "esc to interrupt" line is a RUNNING_MARKER, so it must scroll well
    # out of the last-6-lines "recent" window (see classify_status) before an
    # idle read would be believable -- hence the filler echoes afterward.
    script = (
        "echo start; sleep 1; echo esc to interrupt; sleep 1; "
        "for i in 1 2 3 4 5 6 7 8; do echo filler-$i; done; echo done"
    )
    session = agents.spawn(["bash", "-c", script], cwd="/tmp", label="e2e")
    assert session.startswith(agents.SESSION_PREFIX)

    # First poll: process just started printing "start".
    time.sleep(0.3)
    p0 = agents.poll(session)
    assert "start" in agents.capture(session)

    # Mid-run: "esc to interrupt" marker present -> running.
    time.sleep(1.0)
    p1 = agents.poll(session)
    assert "esc to interrupt" in agents.capture(session)
    assert p1["status"] == "running"

    # After exit: shell has printed "done" and the marker has scrolled out of
    # the recent-lines window. The first poll after the pane stops changing
    # still reads as running (this poll is the one that notices the change,
    # exactly like tmux.rs's own `since_change` bookkeeping), so give it one
    # more poll a full activity window later with no further output -> idle.
    time.sleep(1.0)
    agents.poll(session)
    time.sleep(agents.ACTIVITY_WINDOW_SECS + 1.0)
    p2 = agents.poll(session)
    assert "done" in agents.capture(session)
    assert p2["status"] == "idle"


# --- 2. takeover resync: inject input from outside FRIDAY's code path ------


def test_human_takeover_resync_via_send_keys():
    # Spawn a plain shell -- stands in for a human attaching, typing, and
    # detaching without any cooperation from this module.
    session = agents.spawn(["bash"], cwd="/tmp", label="takeover")
    time.sleep(0.3)
    before = agents.capture(session)
    assert "hello-from-human" not in before

    # This call goes straight through tmux, not through any FRIDAY spawn/
    # write path -- it is exactly what `tmux -L friday attach` + typing +
    # Enter does at the protocol level.
    subprocess.run(
        ["tmux", "-L", agents.SOCKET, "send-keys", "-t", session, "-l", "echo hello-from-human"],
        check=True,
    )
    subprocess.run(["tmux", "-L", agents.SOCKET, "send-keys", "-t", session, "Enter"], check=True)
    time.sleep(0.5)

    after = agents.poll(session)
    assert "hello-from-human" in agents.capture(session)
    assert "hello-from-human" in "\n".join(after["tail"])


# --- 4. isolation: foreign sessions are never selected, killed, or written -

FOREIGN_SESSION = "friday-isolation-canary-do-not-touch"


@pytest.fixture
def foreign_session():
    """A session on the user's *real* (default-socket) tmux server -- what
    this test must prove FRIDAY can never see or touch."""
    _real_tmux(["new-session", "-d", "-s", FOREIGN_SESSION, "sleep 300"])
    yield FOREIGN_SESSION
    _real_tmux(["kill-session", "-t", FOREIGN_SESSION])


def test_foreign_default_socket_session_is_never_enumerated(foreign_session):
    # Confirm it really exists on the default server first.
    check = _real_tmux(["has-session", "-t", foreign_session])
    assert check.returncode == 0

    # The dedicated socket means friday's enumeration is a wholly separate
    # tmux server -- the foreign session cannot appear here at all.
    assert foreign_session not in agents.list_sessions()


def test_foreign_named_session_on_friday_socket_is_refused(foreign_session):
    # Even if something existed on the *friday* socket without the
    # SESSION_PREFIX, the ownership guard must refuse to touch it.
    with pytest.raises(agents.ForeignSessionError):
        agents.capture("some-other-session")
    with pytest.raises(agents.ForeignSessionError):
        agents.send_keys("some-other-session", "rm -rf /", enter=True)
    with pytest.raises(agents.ForeignSessionError):
        agents.stop("some-other-session")


def test_stop_refuses_session_not_actually_live():
    # A friday-prefixed name that was never spawned must still be refused,
    # not silently no-op'd -- kill-session must never be handed a foreign
    # or fabricated target.
    with pytest.raises(agents.ForeignSessionError):
        agents.stop(agents.SESSION_PREFIX + "never-spawned")


# --- 5. permission tier + approval gate ------------------------------------


def test_delegate_and_stop_are_gated_check_is_free():
    assert risk_of("delegate_coding_agent") == Risk.MACHINE_MODIFYING
    assert risk_of("stop_coding_agent") == Risk.DESTRUCTIVE
    assert risk_of("check_agent_status") == Risk.READ_ONLY


@sync
async def test_delegate_coding_agent_denied_without_approval():
    outcome = await execute(
        "id1",
        "delegate_coding_agent",
        {"command": "bash -c 'echo hi'", "cwd": "/tmp"},
        approve=None,
    )
    assert outcome.is_error
    assert "DENIED" in outcome.content
    assert not agents.list_sessions()  # nothing was spawned


@sync
async def test_delegate_coding_agent_runs_when_approved():
    async def approve(_request):
        return True

    outcome = await execute(
        "id2",
        "delegate_coding_agent",
        {"command": "bash -c 'sleep 5'", "cwd": "/tmp", "label": "approved"},
        approve=approve,
    )
    assert not outcome.is_error
    assert "spawned" in outcome.content
    assert agents.list_sessions()


@sync
async def test_stop_coding_agent_denied_without_approval():
    session = agents.spawn(["bash", "-c", "sleep 5"], cwd="/tmp", label="protect")
    outcome = await execute("id3", "stop_coding_agent", {"session": session}, approve=None)
    assert outcome.is_error
    assert "DENIED" in outcome.content
    assert session in agents.list_sessions()  # still alive, refusal was real


@sync
async def test_check_agent_status_auto_executes():
    session = agents.spawn(["bash", "-c", "sleep 5"], cwd="/tmp", label="check")
    time.sleep(0.2)
    outcome = await execute("id4", "check_agent_status", {"session": session}, approve=None)
    assert not outcome.is_error
    assert session in outcome.content
