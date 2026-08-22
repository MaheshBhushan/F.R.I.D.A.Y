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
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from friday import agents as coding_agents
from friday.permissions import RISK, ApprovalCallback, PermissionDenied, Risk, authorize
from friday.tools.sanitize import safe_argv, safe_path
from friday.tools.websearch import web_search

# T9: coding-agent delegation runs real processes in tmux panes, so it sits
# outside the tiers this module used to cover alone. Registered here (a dict
# mutation, not an edit to permissions.py) per that module's own contract --
# RISK is "an allowlist", not something only permissions.py may populate.
#   - check_agent_status: only captures pane text, no side effect -> READ_ONLY.
#   - delegate_coding_agent: spawns a process that can edit the user's files
#     -> MACHINE_MODIFYING (reversible via git / killing the session, but not
#     read-only, so it needs confirmation like install_package).
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
    except asyncio.TimeoutError:
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


async def open_app(app: str) -> str:
    """SAFE_REVERSIBLE. Launch a desktop app (closing it undoes this)."""
    if "/" in app or any(c in app for c in ";&|`$()<>\n"):
        raise PermissionDenied(f"refused: unsafe app name {app!r}")
    binary = shutil.which(app)
    if binary is None:
        return f"[no such app on PATH: {app}]"
    await asyncio.create_subprocess_exec(
        binary,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"launched {app}"


async def install_package(package: str) -> str:
    """MACHINE_MODIFYING. Install a system package (confirm first)."""
    name = package
    if not name.replace("-", "").replace("_", "").replace("+", "").isalnum():
        raise PermissionDenied(f"refused: implausible package name {package!r}")
    return await _run(["pacman", "-S", "--noconfirm", name], timeout=300.0)


async def delete_path(path: str) -> str:
    """DESTRUCTIVE. Delete a file (explicit approval, exact path shown)."""
    resolved = safe_path(path)
    if resolved.is_dir():
        raise PermissionDenied(f"refused: {path!r} is a directory")
    os.remove(resolved)
    return f"deleted {resolved}"


async def delegate_coding_agent(command: str, cwd: str, label: str = "task") -> str:
    """MACHINE_MODIFYING. Spawn a coding agent in a new tmux session (confirm first)."""
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
    """DESTRUCTIVE. Kill a friday-owned agent session (explicit approval)."""
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
}

# Order is load-bearing: `tools` is rendered before `system` and sits inside
# the cached prefix, so this list must be byte-stable across turns.
TOOL_SPECS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read a text file from disk. Confined to the user's home, "
                       "/var/log, /proc, /sys and /tmp; credential files are refused.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~-relative path."},
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
                "lines": {"type": "integer", "description": "How many lines (max 200)."},
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
        "description": "Launch a desktop application by binary name.",
        "input_schema": {
            "type": "object",
            "properties": {"app": {"type": "string"}},
            "required": ["app"],
        },
    },
    {
        "name": "install_package",
        "description": "Install a system package. Modifies the machine; needs confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {"package": {"type": "string"}},
            "required": ["package"],
        },
    },
    {
        "name": "delete_path",
        "description": "Delete a file. Destructive; needs explicit approval.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "delegate_coding_agent",
        "description": "Spawn a coding agent (e.g. `claude`, `codex`) in a new tmux "
                       "session and hand it a task command. Modifies the machine; "
                       "needs confirmation. The user can `tmux -L friday attach -t "
                       "<session>` at any time to watch or take over.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command + args to run in the pane."},
                "cwd": {"type": "string", "description": "Working directory for the agent."},
                "label": {"type": "string", "description": "Short label used in the session name."},
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
        "description": "Kill a friday-owned coding-agent tmux session. Destructive; "
                       "needs explicit approval.",
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
                "num_results": {"type": "integer", "description": "How many results (max 10)."},
            },
            "required": ["query"],
        },
    },
]


@dataclass
class ToolOutcome:
    """Result of one gated tool call, shaped for a tool_result block."""

    tool_use_id: str
    name: str
    content: str
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
    registry: Optional[dict[str, Callable[..., Awaitable[str]]]] = None,
) -> ToolOutcome:
    """Authorize, then run, one tool call. Denials and tool errors both come
    back as `is_error` outcomes so the model can see and recover from them --
    a denial is never silently dropped."""
    registry = TOOLS if registry is None else registry
    try:
        await authorize(name, arguments, approve)
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
