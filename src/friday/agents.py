"""Delegate coding work to real Claude Code / Codex sessions in tmux.

FRIDAY spawns a coding agent inside a tmux pane on a dedicated tmux server
(`tmux -L friday`), writes a task into it, and polls progress by capturing
and diffing pane scrollback -- never by parsing a custom protocol. Because
the agent runs in a plain tmux pane, a human can `tmux -L friday attach -t
<session>`, type, and detach at any time; FRIDAY's next poll just re-reads
the pane, so takeover requires no cooperation from this module at all.

Isolation is two-layered: a dedicated socket (`-L friday`) keeps this code
off the user's real tmux server entirely, and every session FRIDAY creates
is named with `SESSION_PREFIX` so enumeration/kill paths refuse anything
that doesn't match, even on the friday socket. `tmux kill-server` is never
used.

Ported from agent-overlay/src-tauri/src/{tmux.rs,parser.rs}:
  - `classify_status()` is a direct port of `parser.rs::detect_status` (the
    RUNNING_MARKERS / PERMISSION_MARKERS lists and the "scan a wider tail
    for permission dialogs, checked before the running markers" ordering).
  - Session discovery/capture is a much smaller reimplementation of
    `tmux.rs::discover`/`capture_pane`: this module only manages sessions it
    created itself, so the process-tree agent-detection and
    ~/.claude/sessions/<pid>.json per-pid status logic (claude_status.rs) do
    not apply -- FRIDAY already knows which pane is which agent because it
    spawned it. Output-activity tracking (the `since_change` staleness
    window) is reused via `activity_status()` for the same reason
    `parser.rs` needs it: some agents don't print a recognizable spinner.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

SOCKET = "friday"
SESSION_PREFIX = "friday-agent-"
ACTIVITY_WINDOW_SECS = 5.0

# --- ported from parser.rs ---------------------------------------------------

RUNNING_MARKERS = [
    "esc to interrupt",
    "ctrl+c to interrupt",
    "thinking",
    "pondering",
    "working",
    "running…",
    "✻",
    "✳",
    "✽",
]

PERMISSION_MARKERS = [
    "do you want",
    "would you like to",
    "1. yes",
    "(y)es",
    "y/n)",
    "[y/n]",
    "(y/n",
    "apply this change",
    "allow this",
    "approve this",
    "grant permission",
    "waiting for approval",
]


def classify_status(text: str) -> str:
    """"running" | "idle" | "permission", ported from parser.rs::detect_status."""
    lines = text.splitlines()
    trimmed_end = 0
    for i, line in enumerate(lines):
        if line.strip():
            trimmed_end = i + 1
    lines = lines[:trimmed_end]

    permission_zone = [l.lower() for l in lines[-12:]]
    for line in permission_zone:
        if any(m in line for m in PERMISSION_MARKERS):
            return "permission"

    recent = [l.lower() for l in lines[-6:]]
    for line in recent:
        if any(m in line for m in RUNNING_MARKERS):
            return "running"

    return "idle"


# --- session naming / isolation ---------------------------------------------


class ForeignSessionError(Exception):
    """Raised when an operation targets a session FRIDAY did not create."""


def _require_owned(session: str) -> None:
    if not session.startswith(SESSION_PREFIX):
        raise ForeignSessionError(
            f"refused: {session!r} is not a friday-owned session "
            f"(missing {SESSION_PREFIX!r} prefix)"
        )


def _tmux(args: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", "-L", SOCKET, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def new_session_name(label: str = "task") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label) or "task"
    return f"{SESSION_PREFIX}{safe}-{int(time.time() * 1000)}"


# --- lifecycle ----------------------------------------------------------------


@dataclass
class AgentSession:
    session: str
    activity: dict = field(default_factory=dict)  # {"hash": int, "since": float}


def spawn(command: list[str], cwd: str, label: str = "task") -> str:
    """Start `command` in a new detached session on the friday socket.

    Returns the session name. The session name always carries
    `SESSION_PREFIX`, so it is unmistakably friday-owned.
    """
    session = new_session_name(label)
    result = _tmux(["new-session", "-d", "-s", session, "-c", cwd, *command])
    if result.returncode != 0:
        raise RuntimeError(f"failed to spawn session {session!r}: {result.stderr.strip()}")
    # Keep the pane around after the command exits so a finished (or crashed)
    # agent can still be captured/classified instead of the session vanishing
    # out from under a poll.
    _tmux(["set-option", "-t", session, "remain-on-exit", "on"])
    return session


def list_sessions() -> list[str]:
    """Names of all friday-owned sessions on the friday socket."""
    result = _tmux(["list-sessions", "-F", "#{session_name}"])
    if result.returncode != 0:
        return []  # no server running yet, or nothing there
    return [s for s in result.stdout.splitlines() if s.startswith(SESSION_PREFIX)]


def capture(session: str, lines: int = 60) -> str:
    """Capture the last `lines` of visible pane output for `session`."""
    _require_owned(session)
    result = _tmux(["capture-pane", "-p", "-t", session, "-S", f"-{lines}"])
    if result.returncode != 0:
        raise RuntimeError(f"capture-pane failed for {session!r}: {result.stderr.strip()}")
    return result.stdout


def send_keys(session: str, text: str, enter: bool = True) -> None:
    """Write `text` into the session's pane, as a human typing would."""
    _require_owned(session)
    args = ["send-keys", "-t", session, "-l", text] if text else ["send-keys", "-t", session]
    if _tmux(args).returncode != 0:
        raise RuntimeError(f"send-keys failed for {session!r}")
    if enter:
        if _tmux(["send-keys", "-t", session, "Enter"]).returncode != 0:
            raise RuntimeError(f"send Enter failed for {session!r}")


def stop(session: str) -> None:
    """Kill exactly one friday-owned session. Never touches the server."""
    _require_owned(session)
    if session not in list_sessions():
        raise ForeignSessionError(f"refused: {session!r} is not a live friday session")
    result = _tmux(["kill-session", "-t", session])
    if result.returncode != 0:
        raise RuntimeError(f"kill-session failed for {session!r}: {result.stderr.strip()}")


# --- polling / activity tracking ---------------------------------------------

_activity: dict[str, tuple[int, float]] = {}  # session -> (hash, last_change_monotonic)


def poll(session: str, lines: int = 60) -> dict:
    """Capture + classify one session, ported activity-window fallback for
    agents that print no recognizable spinner (parser.rs's `since_change <
    ACTIVITY_WINDOW_SECS` rule, tracked in tmux.rs::discover)."""
    _require_owned(session)
    text = capture(session, lines=lines)
    status = classify_status(text)

    now = time.monotonic()
    h = hash(text)
    prev = _activity.get(session)
    if prev is None or prev[0] != h:
        _activity[session] = (h, now)
        since_change = 0.0
    else:
        since_change = now - prev[1]

    if status == "idle" and since_change < ACTIVITY_WINDOW_SECS:
        status = "running"

    lines = text.splitlines()
    trimmed_end = 0
    for i, line in enumerate(lines):
        if line.strip():
            trimmed_end = i + 1
    return {
        "session": session,
        "status": status,
        "since_change_secs": round(since_change, 1),
        "tail": lines[:trimmed_end][-8:],
    }
