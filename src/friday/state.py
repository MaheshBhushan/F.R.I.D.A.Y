"""World-state snapshot: what is happening on this computer right now.

Polls tmux/git/ss/psutil/proc/sys on demand (1s TTL cache) so the
assistant can answer "what's Codex doing?" from live state instead of
re-investigating from scratch. No background watcher/daemon: callers
poll `snapshot()` whenever they need fresh state.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional dep, degrade gracefully
    psutil = None

_TTL_SECONDS = 1.0
_cache: dict | None = None
_cache_time: float = 0.0

_SECRET_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*)=\S+",
    re.IGNORECASE,
)

# Process names worth calling out explicitly as coding-agent / dev activity.
_NOTABLE_NAMES = {
    "claude",
    "codex",
    "node",
    "python",
    "python3",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
    "vite",
    "next",
    "tsc",
}


def _redact(cmd: str) -> str:
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}=***", cmd)


def _run(argv: list[str], timeout: float = 2.0) -> str | None:
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _popen(argv: list[str]) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return None


def _collect(proc: subprocess.Popen | None, timeout: float = 2.0) -> str | None:
    if proc is None:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.SubprocessError:
        proc.kill()
        return None
    if proc.returncode != 0:
        return None
    return out


def _git_toplevel_proc() -> subprocess.Popen | None:
    return _popen(["git", "rev-parse", "--show-toplevel", "--abbrev-ref", "HEAD"])


def _git_state(toplevel_out: str | None) -> dict | None:
    # `git status --porcelain` needs the repo root from the first call, so
    # it is launched only after that result lands; the other independent
    # subprocesses (tmux, ss) still run concurrently with both git calls.
    if toplevel_out is None:
        return None
    lines = toplevel_out.splitlines()
    if len(lines) < 2:
        return None
    root, branch = lines[0].strip(), lines[1].strip()
    dirty_out = _run(["git", "-C", root, "status", "--porcelain"])
    return {
        "root": root,
        "name": os.path.basename(root),
        "branch": branch or None,
        "dirty": bool(dirty_out and dirty_out.strip()),
    }


def _tmux_state(out: str | None) -> list[dict]:
    if not out:
        return []
    panes = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        session, win_idx, win_name, pane_idx, cmd, path = parts
        panes.append(
            {
                "session": session,
                "window": f"{win_idx}:{win_name}",
                "pane": pane_idx,
                "command": cmd,
                "path": path,
            }
        )
    return panes


def _listening_ports(out: str | None) -> list[dict]:
    if not out:
        return []
    seen = set()
    ports = []
    for line in out.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 5:
            continue
        local_addr = cols[3]
        port = local_addr.rsplit(":", 1)[-1]
        proc_name = None
        m = re.search(r'users:\(\("([^"]+)"', line)
        if m:
            proc_name = m.group(1)
        key = (port, proc_name)
        if key in seen:
            continue
        seen.add(key)
        ports.append({"port": port, "process": proc_name})
    return ports


def _notable_processes() -> list[dict]:
    # Read /proc directly rather than psutil.process_iter(): scanning every
    # process's full info via psutil is ~4x slower than a raw comm/cmdline
    # read for the handful of names we care about.
    procs = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return procs
    for pid in pids:
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
        except OSError:
            continue
        if name not in _NOTABLE_NAMES:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = " ".join(f.read().decode(errors="replace").split("\0")).strip()
        except OSError:
            cmdline = ""
        procs.append({"pid": int(pid), "name": name, "cmdline": _redact(cmdline)})
    return procs


def _resources(project_root: str | None) -> dict:
    res: dict = {}
    try:
        with open("/proc/loadavg") as f:
            res["load_avg"] = [float(x) for x in f.read().split()[:3]]
    except OSError:
        res["load_avg"] = None

    if psutil is not None:
        vm = psutil.virtual_memory()
        res["mem_used_gb"] = round((vm.total - vm.available) / 2**30, 1)
        res["mem_total_gb"] = round(vm.total / 2**30, 1)
    else:
        res["mem_used_gb"] = None
        res["mem_total_gb"] = None

    battery_pct = None
    ac_online = None
    ps_dir = "/sys/class/power_supply"
    try:
        for name in os.listdir(ps_dir):
            type_path = os.path.join(ps_dir, name, "type")
            with open(type_path) as f:
                kind = f.read().strip()
            if kind == "Battery":
                with open(os.path.join(ps_dir, name, "capacity")) as f:
                    battery_pct = int(f.read().strip())
            elif kind == "Mains":
                with open(os.path.join(ps_dir, name, "online")) as f:
                    ac_online = f.read().strip() == "1"
    except OSError:
        pass
    res["battery_pct"] = battery_pct
    res["ac_online"] = ac_online

    disk_free_gb = None
    if project_root:
        try:
            du = os.statvfs(project_root)
            disk_free_gb = round(du.f_bavail * du.f_frsize / 2**30, 1)
        except OSError:
            pass
    res["disk_free_gb"] = disk_free_gb

    return res


def _focused_window() -> str | None:
    # KDE Wayland has no cheap, stable way to query the focused window
    # without a fragile compositor-specific hack (kdotool/D-Bus scripting
    # that breaks across Plasma versions). Skipped per scope.
    return None


def snapshot() -> dict:
    """Return a structured snapshot of current machine/session state.

    Cached for `_TTL_SECONDS` so repeated calls within one turn are free.
    """
    global _cache, _cache_time
    now = time.monotonic()
    if _cache is not None and (now - _cache_time) < _TTL_SECONDS:
        return _cache

    # Launch the independent external commands concurrently so their
    # fork/exec latency overlaps instead of stacking up sequentially.
    git_proc = _git_toplevel_proc()
    tmux_proc = _popen(
        [
            "tmux",
            "list-panes",
            "-a",
            "-F",
            "#{session_name}\t#{window_index}\t#{window_name}\t#{pane_index}\t#{pane_current_command}\t#{pane_current_path}",
        ]
    )
    ss_proc = _popen(["ss", "-tlnp"])

    git_toplevel_out = _collect(git_proc)
    tmux_out = _collect(tmux_proc)
    ss_out = _collect(ss_proc)

    git_state = _git_state(git_toplevel_out)
    snap = {
        "timestamp": time.time(),
        "cwd": os.getcwd(),
        "git": git_state,
        "tmux_panes": _tmux_state(tmux_out),
        "listening_ports": _listening_ports(ss_out),
        "notable_processes": _notable_processes(),
        "resources": _resources(git_state["root"] if git_state else os.getcwd()),
        "focused_window": _focused_window(),
    }

    _cache = snap
    _cache_time = now
    return snap


def summarize(snap: dict) -> str:
    """Compact natural-language-ish summary for LLM-prompt injection."""
    lines = []

    git_state = snap.get("git")
    if git_state:
        dirty = " (dirty)" if git_state["dirty"] else ""
        lines.append(f"Project: {git_state['name']} @ {git_state['branch']}{dirty}")

    panes = snap.get("tmux_panes") or []
    if panes:
        pane_bits = [f"{p['session']}:{p['window']}.{p['pane']}={p['command']}" for p in panes]
        lines.append("tmux panes: " + ", ".join(pane_bits))

    ports = snap.get("listening_ports") or []
    named_ports = [p for p in ports if p["process"]]
    if named_ports:
        port_bits = [f"{p['port']}({p['process']})" for p in named_ports]
        lines.append("Listening: " + ", ".join(port_bits))

    procs = snap.get("notable_processes") or []
    if procs:
        proc_bits = [f"{p['name']}[{p['pid']}]" for p in procs]
        lines.append("Notable procs: " + ", ".join(proc_bits))

    res = snap.get("resources") or {}
    res_bits = []
    if res.get("load_avg"):
        res_bits.append("load " + "/".join(f"{x:.1f}" for x in res["load_avg"]))
    if res.get("mem_used_gb") is not None:
        res_bits.append(f"mem {res['mem_used_gb']}/{res['mem_total_gb']}GB")
    if res.get("battery_pct") is not None:
        ac = "AC" if res.get("ac_online") else "battery"
        res_bits.append(f"batt {res['battery_pct']}% ({ac})")
    if res.get("disk_free_gb") is not None:
        res_bits.append(f"disk free {res['disk_free_gb']}GB")
    if res_bits:
        lines.append("Resources: " + ", ".join(res_bits))

    if not lines:
        return "No notable state detected."
    return " | ".join(lines)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print a world-state snapshot.")
    parser.add_argument("--dump", action="store_true", help="print JSON + summary")
    args = parser.parse_args()

    if not args.dump:
        parser.print_help()
        return

    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0
    t0 = time.perf_counter()
    snap = snapshot()
    cold_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    snap2 = snapshot()
    warm_cached_ms = (time.perf_counter() - t1) * 1000

    time.sleep(_TTL_SECONDS + 0.05)
    t2 = time.perf_counter()
    snap3 = snapshot()
    warm_uncached_ms = (time.perf_counter() - t2) * 1000

    print(json.dumps(snap, indent=2))
    print()
    print(summarize(snap))
    print()
    print(f"cold call: {cold_ms:.2f}ms")
    print(f"cached call (within TTL): {warm_cached_ms:.2f}ms")
    print(f"fresh call after TTL expiry: {warm_uncached_ms:.2f}ms")


if __name__ == "__main__":
    _main()
