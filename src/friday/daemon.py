"""Process lifecycle for the FRIDAY daemon: start, stop, and honest liveness.

Three problems this solves, none of which a bare PID file handles.

**Who is in charge.** The systemd unit sets `Restart=always`. If `friday stop`
signalled the process directly while systemd owned it, systemd would restart it
within three seconds and the command would look broken. So every lifecycle
operation first asks who the supervisor is, and delegates when the answer is
systemd.

**What "running" means.** A PID that exists is not a working assistant. The
authoritative check is the gateway answering `health`; the PID is only a
fallback for when the gateway is disabled. This matters because the interesting
failure -- wedged, holding the microphone, answering nothing -- looks perfectly
alive to a PID check.

**PID recycling.** A stale PID file after a hard reboot can name a process that
now belongs to something else entirely, and signalling it would kill an
innocent bystander. Every PID read is verified against the process's own
command line before anything is sent to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

FRIDAY_HOME = Path(os.environ.get("FRIDAY_HOME", Path.home() / ".friday"))
PID_PATH = FRIDAY_HOME / "friday.pid"
LOG_PATH = FRIDAY_HOME / "friday.log"
UNIT_NAME = "friday.service"
ENV_PATH = FRIDAY_HOME / "env"
# The installed package root: .../friday/src/friday/daemon.py -> .../friday
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Cold start loads the wake-word ONNX model and opens a capture stream, so the
# first health answer is seconds away, not milliseconds. Waiting less than this
# would report a false failure for a daemon that was merely still booting.
START_TIMEOUT_S = 45.0
# SIGINT unwinds the capture stream and releases the microphone. Anything that
# takes longer than this to do that is stuck, not busy.
STOP_TIMEOUT_S = 15.0
HEALTH_TIMEOUT_S = 3.0


class Supervisor(str, Enum):
    """Who owns the process lifecycle."""

    SYSTEMD = "systemd"
    DIRECT = "direct"


@dataclass
class Status:
    running: bool
    healthy: bool
    pid: Optional[int]
    supervisor: Supervisor
    health: Optional[dict] = None
    detail: str = ""


# --------------------------------------------------------------- supervisor

def _systemctl(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def unit_installed() -> bool:
    """True when the user unit exists, whether or not it is enabled."""
    try:
        result = _systemctl("cat", UNIT_NAME, timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def supervisor() -> Supervisor:
    """Decide who to route lifecycle commands through.

    The unit merely *existing* is enough to delegate. An installed-but-stopped
    unit still means the user's intent is systemd, and starting a competing
    bare process would give them two daemons fighting over one microphone --
    with only one of them visible to `systemctl status`.
    """
    return Supervisor.SYSTEMD if unit_installed() else Supervisor.DIRECT


# ---------------------------------------------------------------- pid file

def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _is_friday(pid: int) -> bool:
    """Guard against PID recycling before signalling anything.

    A stale pidfile surviving a reboot can name an unrelated process, and
    `kill` does not ask whether you meant it.
    """
    line = _cmdline(pid)
    return "friday" in line and ("-m friday" in line or "friday.core" in line)


def read_pid() -> Optional[int]:
    """The recorded PID, if it names a live FRIDAY. Cleans up if it does not."""
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _clear_pid()
        return None
    except PermissionError:
        # Alive but owned by another user -- not ours to manage, and certainly
        # not ours to signal.
        return None
    if not _is_friday(pid):
        _clear_pid()
        return None
    return pid


def _write_pid(pid: int) -> None:
    FRIDAY_HOME.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(f"{pid}\n")


def _clear_pid() -> None:
    with contextlib.suppress(OSError):
        PID_PATH.unlink()


# ------------------------------------------------------------------ health

def health(timeout: float = HEALTH_TIMEOUT_S) -> Optional[dict]:
    """Ask the gateway how it is. None means "did not answer", not "unhealthy".

    Deliberately returns None rather than raising: every caller here treats an
    unreachable gateway as one signal among several, not an error.
    """
    from friday.gateway.client import GatewayClient

    async def _probe() -> Optional[dict]:
        client = GatewayClient()
        try:
            await asyncio.wait_for(client.open(), timeout=timeout)
            reply = await client.connect(name="friday-cli")
            if not reply.get("ok"):
                return {"ok": False, "error": reply.get("error", {})}
            reply = await client.request("health", timeout=timeout)
            return reply.get("result") if reply.get("ok") else None
        except Exception:  # noqa: BLE001 - unreachable is a normal answer here
            return None
        finally:
            await client.close()

    try:
        return asyncio.run(_probe())
    except Exception:  # noqa: BLE001
        return None


def _systemd_pid() -> Optional[int]:
    result = _systemctl("show", UNIT_NAME, "--property=MainPID", "--value")
    if result.returncode != 0:
        return None
    try:
        pid = int(result.stdout.strip())
    except ValueError:
        return None
    return pid or None


def status() -> Status:
    """Best available answer to "is FRIDAY up, and is she well?"."""
    who = supervisor()
    info = health()

    if who is Supervisor.SYSTEMD:
        active = _systemctl("is-active", UNIT_NAME).stdout.strip()
        pid = _systemd_pid()
        running = active == "active"
        return Status(
            running=running,
            healthy=bool(info and info.get("ok")),
            pid=pid,
            supervisor=who,
            health=info,
            detail=active,
        )

    pid = read_pid()
    running = pid is not None
    return Status(
        running=running or bool(info),
        healthy=bool(info and info.get("ok")),
        pid=pid,
        supervisor=who,
        health=info,
        detail="running" if running else "stopped",
    )


# ------------------------------------------------------------------- start

def start(*, wait: bool = True, timeout: float = START_TIMEOUT_S) -> Status:
    """Bring the daemon up, or return the status of the one already running.

    Idempotent on purpose: `friday start` twice must not produce two daemons
    contending for the microphone.
    """
    existing = status()
    if existing.running:
        existing.detail = "already running"
        return existing

    who = supervisor()
    if who is Supervisor.SYSTEMD:
        result = _systemctl("start", UNIT_NAME, timeout=timeout + 10)
        if result.returncode != 0:
            return Status(False, False, None, who,
                          detail=(result.stderr or result.stdout).strip()
                                 or "systemctl start failed")
    else:
        _spawn()

    if not wait:
        return status()
    return _await_healthy(timeout)


def _spawn() -> int:
    """Launch a detached daemon with its output captured to the log file.

    `start_new_session=True` puts it in its own session and process group, so
    it survives the terminal closing and a Ctrl-C in the parent shell does not
    reach it. Without that, `friday start` would create a daemon that dies with
    the window that started it.
    """
    FRIDAY_HOME.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "ab", buffering=0)
    log.write(f"\n=== friday start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
              .encode())
    env = dict(os.environ)
    # Load credentials ourselves rather than inheriting whatever the invoking
    # shell happened to have. Requiring `source ~/.friday/env` before
    # `friday start` is a footgun: the daemon comes up, reports healthy, and
    # the voice loop is silently down -- which is exactly what happened the
    # first time this was run from a fresh terminal. Existing values win, so an
    # explicit override on the command line still takes precedence.
    for key, value in load_env_file().items():
        env.setdefault(key, value)
    # No tty for a detached daemon, so the ANSI status line has nothing to draw
    # on; state still reaches the status file and the gateway's events.
    env.setdefault("FRIDAY_INDICATOR", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "friday"],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, env=env,
        # Pinned, and pinned to the same directory the systemd unit uses.
        # State queries like "what branch am i on" are answered relative to the
        # daemon's cwd, so inheriting the launching shell's directory made the
        # answer depend on where you happened to be standing -- and starting
        # from $HOME put her outside any repo at all. Behaviour must not differ
        # between `friday start` and `systemctl start`.
        cwd=str(PROJECT_ROOT),
    )
    _write_pid(proc.pid)
    return proc.pid


def _await_healthy(timeout: float) -> Status:
    """Poll until the gateway answers, the process dies, or time runs out.

    Bails early on a dead process rather than burning the full timeout: a
    daemon that exited during boot is never going to answer, and making the
    user wait 45s to learn that wastes the one thing they came for.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = status()
        if current.healthy:
            return current
        if current.supervisor is Supervisor.DIRECT and read_pid() is None:
            return Status(False, False, None, current.supervisor,
                          detail="process exited during startup; see the log")
        time.sleep(0.4)
    final = status()
    final.detail = f"did not become healthy within {timeout:.0f}s"
    return final


# -------------------------------------------------------------------- stop

def stop(*, timeout: float = STOP_TIMEOUT_S) -> tuple[bool, str]:
    """Shut the daemon down gracefully, escalating only if it refuses.

    SIGINT first, always. The app's handler unwinds the capture stream and
    releases the microphone; a hard kill can leave the echo-cancel module
    loaded, and a leaked virtual source renumbers every other application's
    device list. That is the bug that silently broke Chrome for nine days.
    """
    who = supervisor()
    if who is Supervisor.SYSTEMD:
        result = _systemctl("stop", UNIT_NAME, timeout=timeout + 10)
        if result.returncode != 0:
            return False, (result.stderr or result.stdout).strip() or "stop failed"
        return True, "stopped (systemd)"

    pid = read_pid()
    if pid is None:
        _clear_pid()
        return True, "not running"

    for sig, label in ((signal.SIGINT, "SIGINT"),
                       (signal.SIGTERM, "SIGTERM"),
                       (signal.SIGKILL, "SIGKILL")):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            _clear_pid()
            return True, "stopped"
        if _wait_gone(pid, timeout if sig is signal.SIGINT else 5.0):
            _clear_pid()
            note = "stopped" if sig is signal.SIGINT else f"stopped after {label}"
            if sig is not signal.SIGINT:
                # A daemon that did not unwind cleanly never ran its own
                # cleanup, so do it here. Telling the user to go and unload a
                # module by hand is not a fix: a leaked echo-cancel source
                # renumbers every application's capture device list, and the
                # symptom shows up hours later in an unrelated app with no
                # visible connection to FRIDAY. Repair it now, while the cause
                # is still obvious.
                freed = reap_echo_cancel()
                if freed:
                    note += f" (cleaned up {freed} leaked echo-cancel module"
                    note += "s)" if freed > 1 else ")"
            return True, note

    return False, f"pid {pid} survived SIGKILL"


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.15)
    return False


def restart(*, timeout: float = START_TIMEOUT_S) -> Status:
    ok, note = stop()
    if not ok:
        return Status(False, False, None, supervisor(), detail=f"stop failed: {note}")
    return start(timeout=timeout)


# ------------------------------------------------------- crash-path cleanup

def reap_echo_cancel() -> int:
    """Unload any leftover echo-cancel module. Returns how many were freed.

    Only ever called when FRIDAY did not exit cleanly. Safe to call when there
    is nothing to do, and deliberately narrow: it matches the module by name
    rather than tracking an id, because the id is lost precisely in the case
    that matters -- the process was killed before it could record anything.

    The blast radius is acceptable because FRIDAY is the only thing on this
    machine that loads module-echo-cancel at runtime; a config-loaded one lives
    in the PipeWire config directory and is not visible to `pactl unload`.
    """
    try:
        listing = subprocess.run(["pactl", "list", "short", "modules"],
                                 capture_output=True, text=True,
                                 timeout=5.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0
    if listing.returncode != 0:
        return 0

    freed = 0
    for line in listing.stdout.splitlines():
        if "module-echo-cancel" not in line:
            continue
        module_id = line.split("\t", 1)[0].strip()
        if not module_id.isdigit():
            continue
        try:
            result = subprocess.run(["pactl", "unload-module", module_id],
                                    capture_output=True, timeout=5.0, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            freed += 1
    return freed


# ------------------------------------------------------------ env file

def load_env_file(path: Optional[Path] = None) -> dict[str, str]:
    """Parse `~/.friday/env` into a dict. Missing file is not an error.

    Parsed in Python rather than sourced through a shell. Sourcing would handle
    exotic syntax, but it also executes whatever is in the file -- and this file
    exists specifically to hold API keys, so it is the last thing that should
    gain the ability to run commands as a side effect of starting the daemon.

    Understands the format the file actually uses: `export KEY=value`, optional
    surrounding quotes, `#` comments, blank lines. Anything it does not
    understand is skipped rather than guessed at.
    """
    target = path or ENV_PATH
    try:
        raw = target.read_text()
    except OSError:
        return {}

    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values
