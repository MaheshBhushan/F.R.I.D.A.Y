"""The tool surface FRIDAY exposes to the LLM.

Five read-mostly tools plus one machine-modifying and one destructive tool,
each with a tier in `friday.permissions.RISK`, plus T9's coding-agent
delegation tools (one read-only, one machine-modifying, one destructive).
`TOOL_SPECS` is the Anthropic tool-definition list (stable ordering -- it is
part of the cached prefix, so the order must not vary between turns).

Execution goes through `execute()`, which authorizes first, then runs. Tool
arguments are sanitised inside each tool by `friday.tools.sanitize`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from friday import agents as coding_agents
from friday.permissions import RISK, ApprovalCallback, PermissionDenied, Risk, authorize
from friday.tools.sanitize import safe_argv, safe_path
# Imported as a module, not `from ... import screenshot`: the function
# would shadow the `friday.tools.screenshot` submodule attribute.
from friday.tools import screenshot as _screenshot
from friday.tools.websearch import web_search
from friday.tools.forecast import SPEC as FORECAST_SPEC, query_forecast

# T9: coding-agent delegation runs real processes in tmux panes, so it sits
# outside the tiers this module used to cover alone. Registered here (a dict
# mutation, not an edit to permissions.py) per that module's own contract --
# RISK is "an allowlist", not something only permissions.py may populate.
#   - check_agent_status: only captures pane text, no side effect -> READ_ONLY.
#   - delegate_coding_agent: spawns a process that can edit the user's files
#     -> MACHINE_MODIFYING (reversible via git / killing the session, but not
#     read-only).
#   - stop_coding_agent: kills an in-progress agent, which can discard
#     uncommitted work the agent was mid-way through -> DESTRUCTIVE.
RISK.update(
    {
        "check_agent_status": Risk.READ_ONLY,
        "delegate_coding_agent": Risk.MACHINE_MODIFYING,
        "stop_coding_agent": Risk.DESTRUCTIVE,
    }
)

# web_search is the first tool that sends data OUT to a third party (Exa),
# but the call itself has no side effect on this machine and returns
# read-only search results -> READ_ONLY, auto-execute. The exfiltration risk
# is handled by `safe_query()` refusing secret-shaped or local-data queries,
# not by an approval prompt (a voice assistant that confirms every search is
# useless).
RISK.update({"web_search": Risk.READ_ONLY})

# screenshot reads the display and ships it to the model. Nothing on the
# machine changes -> READ_ONLY. The privacy question is answered by the tool
# description (only for questions about the screen) and FRIDAY_SCREENSHOT=0,
# not by a per-call prompt.
RISK.update({"screenshot": Risk.READ_ONLY})
RISK.update({"query_forecast": Risk.READ_ONLY})

# Tools whose results carry local file content into the conversation, and
# tools that send conversation content to a third party. Once a turn has used
# a source, the sinks stop auto-executing (see `execute(tainted=...)`).
TAINT_SOURCES = frozenset({"read_file"})
EXFIL_SINKS = frozenset({"web_search", "screenshot"})

_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")

MAX_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT = 5.0


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[clipped, {len(text)} chars total]"


async def _run(argv: list[str], timeout: float = COMMAND_TIMEOUT) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.communicate(), timeout=1.0)
        raise
    except TimeoutError:
        proc.kill()
        return f"[timed out after {timeout}s]"
    return _clip(out.decode(errors="replace"))


# --- tool implementations ----------------------------------------------------


async def read_file(path: str, max_lines: int = 200) -> str:
    """READ_ONLY. Read a text file, confined to the readable roots."""
    resolved = safe_path(path)
    if not resolved.is_file():
        return f"[not a readable file: {path}]"
    lines = []
    with resolved.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                lines.append(f"...[truncated at {max_lines} lines]")
                break
            lines.append(line.rstrip("\n"))
    return _clip("\n".join(lines))


async def list_processes(name: Optional[str] = None) -> str:
    """READ_ONLY. Inspect running processes, optionally filtered by name."""
    argv = ["ps", "-eo", "pid,pcpu,pmem,etime,comm,args", "--sort=-pcpu"]
    out = await _run(argv)
    if not name:
        return _clip("\n".join(out.splitlines()[:40]))
    header, *rows = out.splitlines()
    hits = [r for r in rows if name.lower() in r.lower()]
    return _clip("\n".join([header, *hits[:40]]))


async def read_log(unit: Optional[str] = None, lines: int = 50) -> str:
    """READ_ONLY. Tail a systemd journal unit, or the user journal."""
    lines = max(1, min(int(lines), 200))
    argv = ["journalctl", "--no-pager", "-n", str(lines)]
    if unit:
        if not _UNIT_RE.match(unit):
            raise PermissionDenied(f"refused: implausible unit name {unit!r}")
        argv += ["-u", unit]
    return await _run(argv)


async def run_readonly_command(command: str) -> str:
    """READ_ONLY. Run one allowlisted, non-mutating command with shell=False."""
    return await _run(safe_argv(command))


# open_app auto-executes (SAFE_REVERSIBLE), so it launches only programs that
# an installed .desktop entry presents as a graphical app. Anything else --
# power/session control, shells, interpreters, arbitrary binaries on PATH --
# is refused. The denylist stays as a second layer for odd .desktop entries.
OPEN_APP_DENIED = frozenset({
    "reboot", "poweroff", "halt", "shutdown", "systemctl", "loginctl",
    "init", "telinit", "kexec", "suspend", "hibernate", "zzz",
    "xdg-screensaver", "pkill", "killall", "kill", "sudo", "doas", "su",
    "pkexec", "rm", "dd", "mkfs", "wipefs", "sh", "bash", "zsh", "fish",
    "dash", "python", "python3", "node", "bun", "perl", "ruby", "env",
})


def _application_dirs() -> list[Path]:
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(d) / "applications" for d in [data_home, *data_dirs.split(":")] if d]


def _parse_desktop_entry(path: Path) -> Optional[str]:
    """The Exec program of a launchable, visible, non-terminal Application entry."""
    fields: dict[str, str] = {}
    in_entry = False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_entry = line == "[Desktop Entry]"
            continue
        if in_entry and "=" in line:
            key, _, value = line.partition("=")
            fields.setdefault(key.strip(), value.strip())
    if fields.get("Type") != "Application" or not fields.get("Exec"):
        return None
    if any(fields.get(k, "").lower() == "true" for k in ("NoDisplay", "Hidden", "Terminal")):
        return None
    try:
        argv = shlex.split(fields["Exec"])
    except ValueError:
        return None
    argv = [a for a in argv if "=" not in a or a.startswith("/")]
    if argv and Path(argv[0]).name == "env":
        argv = argv[1:]
    return argv[0] if argv else None


def desktop_apps() -> dict[str, str]:
    """Map launchable names (desktop-file id and program name) to the program."""
    apps: dict[str, str] = {}
    for appdir in reversed(_application_dirs()):  # earlier XDG dirs win
        if not appdir.is_dir():
            continue
        for entry in appdir.rglob("*.desktop"):
            program = _parse_desktop_entry(entry)
            if program is None:
                continue
            apps[entry.stem] = program
            apps[Path(program).name] = program
    return apps


async def open_app(app: str) -> str:
    """SAFE_REVERSIBLE. Launch a desktop app (closing it undoes this)."""
    if "/" in app or any(c in app for c in ";&|`$()<>\n"):
        raise PermissionDenied(f"refused: unsafe app name {app!r}")
    program = (await asyncio.to_thread(desktop_apps)).get(app)
    if program is None:
        raise PermissionDenied(f"refused: {app!r} is not an installed desktop app")
    name = Path(program).name
    if name in OPEN_APP_DENIED or name.startswith(("systemd-", "mkfs.")):
        raise PermissionDenied(f"refused: {app!r} is not a reversible desktop app")
    binary = program if Path(program).is_absolute() else shutil.which(program)
    if binary is None or not os.access(binary, os.X_OK):
        return f"[no such app on PATH: {app}]"
    await asyncio.create_subprocess_exec(
        binary,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"launched {app}"


async def install_package(package: str) -> str:
    """MACHINE_MODIFYING. Install a system package."""
    name = package
    if not name.replace("-", "").replace("_", "").replace("+", "").isalnum():
        raise PermissionDenied(f"refused: implausible package name {package!r}")
    return await _run(["sudo", "-n", "pacman", "-S", "--noconfirm", name], timeout=300.0)


async def delete_path(path: str) -> str:
    """DESTRUCTIVE. Delete a file."""
    resolved = safe_path(path)
    if resolved.is_dir():
        raise PermissionDenied(f"refused: {path!r} is a directory")
    os.remove(resolved)
    return f"deleted {resolved}"


async def delegate_coding_agent(command: str, cwd: str, label: str = "task") -> str:
    """MACHINE_MODIFYING. Spawn a coding agent in a new tmux session."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise PermissionDenied(f"refused: unparseable command {command!r} ({exc})")
    if not argv:
        raise PermissionDenied("refused: empty command")
    session = await asyncio.to_thread(coding_agents.spawn, argv, cwd, label)
    return f"spawned {session}"


async def check_agent_status(session: str) -> str:
    """READ_ONLY. Capture a friday-owned agent session's pane and classify it."""
    result = await asyncio.to_thread(coding_agents.poll, session)
    return _clip(
        f"session={result['session']} status={result['status']} "
        f"since_change={result['since_change_secs']}s\n" + "\n".join(result["tail"])
    )


async def stop_coding_agent(session: str) -> str:
    """DESTRUCTIVE. Kill a friday-owned agent session."""
    await asyncio.to_thread(coding_agents.stop, session)
    return f"stopped {session}"


# --- registry ---------------------------------------------------------------

TOOLS: dict[str, Callable[..., Awaitable[str]]] = {
    "read_file": read_file,
    "list_processes": list_processes,
    "read_log": read_log,
    "run_readonly_command": run_readonly_command,
    "open_app": open_app,
    "install_package": install_package,
    "delete_path": delete_path,
    "delegate_coding_agent": delegate_coding_agent,
    "check_agent_status": check_agent_status,
    "stop_coding_agent": stop_coding_agent,
    "web_search": web_search,
    "screenshot": _screenshot.screenshot,
    "query_forecast": query_forecast,
}

# Order is load-bearing: `tools` is rendered before `system` and sits inside
# the cached prefix, so this list must be byte-stable across turns.
TOOL_SPECS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read a text file from disk. Confined to ~/Documents, "
        "~/Downloads, ~/Desktop, ~/notes, ~/projects, ~/code, ~/src, ~/workspace, "
        "FRIDAY_READ_ROOTS, /var/log, /proc, /sys and /tmp; hidden files and "
        "credential files are refused.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or ~-relative path.",
                },
                "max_lines": {"type": "integer", "description": "Max lines to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_processes",
        "description": "List running processes sorted by CPU, optionally filtered by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Substring filter."}},
        },
    },
    {
        "name": "read_log",
        "description": "Tail the systemd journal, optionally for one unit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "description": "systemd unit name."},
                "lines": {
                    "type": "integer",
                    "description": "How many lines (max 200).",
                },
            },
        },
    },
    {
        "name": "run_readonly_command",
        "description": "Run one read-only shell command (ls, ps, df, free, uptime, "
        "uname, git status/log/diff, systemctl status, ...). No pipes, "
        "redirection, or command substitution.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "open_app",
        "description": "Launch an installed desktop application (one with a "
        ".desktop entry) by its program name or desktop-file id.",
        "input_schema": {
            "type": "object",
            "properties": {"app": {"type": "string"}},
            "required": ["app"],
        },
    },
    {
        "name": "install_package",
        "description": "Install a system package.",
        "input_schema": {
            "type": "object",
            "properties": {"package": {"type": "string"}},
            "required": ["package"],
        },
    },
    {
        "name": "delete_path",
        "description": "Delete a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "delegate_coding_agent",
        "description": "Spawn a coding agent (e.g. `claude`, `codex`) in a new tmux "
        "session and hand it a task command. The user can `tmux -L friday attach -t "
        "<session>` at any time to watch or take over.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command + args to run in the pane.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the agent.",
                },
                "label": {
                    "type": "string",
                    "description": "Short label used in the session name.",
                },
            },
            "required": ["command", "cwd"],
        },
    },
    {
        "name": "check_agent_status",
        "description": "Capture a friday-owned coding-agent tmux session's pane and "
        "classify it as running, idle, or waiting for approval.",
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "stop_coding_agent",
        "description": "Kill a friday-owned coding-agent tmux session.",
        "input_schema": {
            "type": "object",
            "properties": {"session": {"type": "string"}},
            "required": ["session"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web via Exa and get back a short ranked list of "
        "titles, URLs and snippets. Refuses queries that look like "
        "credentials or local data rather than a search phrase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search phrase."},
                "num_results": {
                    "type": "integer",
                    "description": "How many results (max 10).",
                },
            },
            "required": ["query"],
        },
    },
    _screenshot.SPEC,
    FORECAST_SPEC,
]


@dataclass
class ToolOutcome:
    """Result of one gated tool call, shaped for a tool_result block."""

    tool_use_id: str
    name: str
    # A string, or a list of content blocks (text/image) for tools whose
    # result is not text -- `screenshot` returns an image block.
    content: "str | list[dict]"
    is_error: bool = False

    def to_result_block(self) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


async def execute(
    tool_use_id: str,
    name: str,
    arguments: dict,
    *,
    approve: Optional[ApprovalCallback] = None,
    registry: Optional[dict[str, Callable[..., Awaitable["str | list[dict]"]]]] = None,
    tainted: bool = False,
) -> ToolOutcome:
    """Authorize, then run, one tool call. Denials and tool errors both come
    back as `is_error` outcomes so the model can see and recover from them --
    a denial is never silently dropped.

    `tainted` means this turn already read local file content. Tools that send
    data off the machine (`EXFIL_SINKS`) then need approval, so text planted in
    a file cannot quietly ship that file out through a search query or a
    screenshot."""
    registry = TOOLS if registry is None else registry
    try:
        await authorize(name, arguments, approve,
                        force_approval=tainted and name in EXFIL_SINKS)
        fn = registry.get(name)
        if fn is None:
            raise PermissionDenied(f"unrecognised tool {name!r}: no implementation")
        content = await fn(**arguments)
    except PermissionDenied as exc:
        return ToolOutcome(tool_use_id, name, f"DENIED: {exc}", is_error=True)
    except TypeError as exc:
        return ToolOutcome(tool_use_id, name, f"BAD ARGUMENTS: {exc}", is_error=True)
    except Exception as exc:  # tool failures are data, not crashes
        return ToolOutcome(tool_use_id, name, f"ERROR: {type(exc).__name__}: {exc}", is_error=True)
    return ToolOutcome(tool_use_id, name, content)
