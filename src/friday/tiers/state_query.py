"""T7: sub-100ms state-query answerer.

Answers Tier.STATE_QUERY questions directly from T5's `friday.state`
snapshot - no LLM call, no network I/O beyond what snapshot() already
does. Ladder rung 6: a dict of question-kind -> formatter function over
the snapshot dict, built with plain f-strings. No query DSL, no
template engine, no NLG library.

If a question kind cannot be answered from the snapshot (e.g. test/CI
failure status, which the snapshot does not track), the answer says so
honestly and `escalate` is set so the caller can hand off to the
reasoning tier instead of fabricating.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from friday import state as state_mod
from friday.core.spans import TurnSpan

# Names T5 labels "notable" that we treat as coding-agent sessions rather
# than generic dev tooling, for the collapsed speech phrasing.
_AGENT_NAMES = ("claude", "codex")


# Reuse the router's normalizer rather than keeping a second copy: the two
# drifted apart once already (contracted vs expanded "what is"), which routed
# utterances here that this module then could not match.
from friday.router import _normalize


# --- question-kind patterns --------------------------------------------------
# Superset of router.py's Tier 2 shapes plus a few resource-pressure phrasings
# from T7's own scope; router.py decides *whether* something is routed here,
# this table decides *what kind* of state question it is once it arrives.

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("agent_doing", re.compile(r"^what(?:'s| is)\s+(\S+)\s+doing$")),
    ("whats_running", re.compile(r"^what'?s\s+running$")),
    ("whats_happening", re.compile(r"^what'?s\s+happening$")),
    ("in_progress", re.compile(r"^what'?s\s+in\s+progress$")),
    ("branch", re.compile(r"^what\s+branch\s+(?:am i on|is this)$")),
    ("is_running", re.compile(r"^is\s+(.+?)\s+running$")),
    ("failures", re.compile(r"^any\s+(?:failures|errors)$")),
    ("test_failures", re.compile(r"^any\s+tests?\s+fail(?:ing|ed)$")),
    ("test_failures_count", re.compile(r"^how many\s+tests?\s+fail(?:ed|ing)$")),
    (
        "resources",
        re.compile(
            r"^(?:how'?s|what'?s)\s+(?:the\s+)?(?:load|memory|ram|battery|disk)"
            r"(?:\s+(?:usage|looking|at))?$"
        ),
    ),
)


@dataclass
class StateAnswer:
    """Result of answer(). `escalate` is True whenever the snapshot could
    not honestly support the question, so the caller can hand off to the
    reasoning tier instead of guessing."""

    text: str
    answerable: bool
    escalate: bool
    kind: Optional[str]


def _collapse_procs(procs: list[dict]) -> list[str]:
    """Speech-friendly counts, e.g. ["four Claude sessions", "three Codex
    sessions", "two node processes"] instead of T5's flat name[pid] list,
    which is noise when spoken aloud."""
    counts = Counter(p["name"] for p in procs)
    bits = []
    for name in _AGENT_NAMES:
        if counts.get(name):
            n = counts[name]
            plural = "s" if n != 1 else ""
            bits.append(f"{n} {name.capitalize()} session{plural}")
    for name in sorted(counts):
        if name in _AGENT_NAMES:
            continue
        n = counts[name]
        plural = "es" if n != 1 else ""
        bits.append(f"{n} {name} process{plural}")
    return bits


def _dirty_suffix(git: Optional[dict]) -> str:
    if git and git.get("dirty"):
        return " (dirty)"
    return ""


def _fmt_branch(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    git = snap.get("git")
    if not git:
        return StateAnswer(
            "I can't tell - the current directory doesn't look like a git repo.",
            answerable=False,
            escalate=True,
            kind="branch",
        )
    return StateAnswer(
        f"You're on {git['branch']} in {git['name']}{_dirty_suffix(git)}.",
        answerable=True,
        escalate=False,
        kind="branch",
    )


def _fmt_agent_doing(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    target = match.group(1).lower() if match else ""
    panes = snap.get("tmux_panes") or []
    hits = [p for p in panes if target in p["command"].lower() or target in p["session"].lower()]
    if hits:
        bits = [f"{p['session']} ({p['command']}) in {p['path']}" for p in hits]
        return StateAnswer(
            f"{target.capitalize()} is running: " + "; ".join(bits) + ".",
            answerable=True,
            escalate=False,
            kind="agent_doing",
        )
    procs = [p for p in (snap.get("notable_processes") or []) if target in p["name"].lower()]
    if procs:
        return StateAnswer(
            f"{target.capitalize()} has {len(procs)} process(es) running, but no tmux pane shows what it's doing.",
            answerable=True,
            escalate=False,
            kind="agent_doing",
        )
    return StateAnswer(
        f"I don't see {target} running anywhere right now - no tmux pane or process for it.",
        answerable=False,
        escalate=True,
        kind="agent_doing",
    )


def _fmt_whats_running(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    bits = _collapse_procs(snap.get("notable_processes") or [])
    ports = [p for p in (snap.get("listening_ports") or []) if p["process"]]
    if not bits and not ports:
        return StateAnswer(
            "Nothing notable running right now.",
            answerable=True,
            escalate=False,
            kind="whats_running",
        )
    parts = []
    if bits:
        parts.append(", ".join(bits))
    if ports:
        port_bits = [f"{p['process']} on {p['port']}" for p in ports]
        parts.append("listening: " + ", ".join(port_bits))
    return StateAnswer(
        "; ".join(parts) + ".",
        answerable=True,
        escalate=False,
        kind="whats_running",
    )


def _fmt_whats_happening(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    git = snap.get("git")
    running = _fmt_whats_running(snap, None)
    parts = []
    if git:
        parts.append(f"On {git['name']} @ {git['branch']}{_dirty_suffix(git)}")
    parts.append(running.text.rstrip("."))
    return StateAnswer(
        ". ".join(parts) + ".",
        answerable=True,
        escalate=False,
        kind="whats_happening",
    )


def _fmt_in_progress(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    git = snap.get("git")
    panes = snap.get("tmux_panes") or []
    parts = []
    if git and git.get("dirty"):
        parts.append(f"{git['name']} has uncommitted changes on {git['branch']}")
    if panes:
        pane_bits = [f"{p['session']} running {p['command']}" for p in panes]
        parts.append(", ".join(pane_bits))
    if not parts:
        return StateAnswer(
            "Nothing appears to be in progress right now.",
            answerable=True,
            escalate=False,
            kind="in_progress",
        )
    return StateAnswer(
        "; ".join(parts) + ".",
        answerable=True,
        escalate=False,
        kind="in_progress",
    )


def _fmt_is_running(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    target = match.group(1).lower() if match else ""
    ports = snap.get("listening_ports") or []
    panes = snap.get("tmux_panes") or []
    procs = snap.get("notable_processes") or []
    port_hit = next((p for p in ports if p["process"] and target in p["process"].lower()), None)
    pane_hit = next((p for p in panes if target in p["command"].lower()), None)
    proc_hit = next((p for p in procs if target in p["name"].lower()), None)
    if port_hit:
        return StateAnswer(
            f"Yes - {port_hit['process']} is listening on port {port_hit['port']}.",
            answerable=True,
            escalate=False,
            kind="is_running",
        )
    if pane_hit:
        return StateAnswer(
            f"Yes - {pane_hit['session']} has a pane running {pane_hit['command']}.",
            answerable=True,
            escalate=False,
            kind="is_running",
        )
    if proc_hit:
        return StateAnswer(
            f"Yes - {proc_hit['name']} is running (pid {proc_hit['pid']}).",
            answerable=True,
            escalate=False,
            kind="is_running",
        )
    return StateAnswer(
        f"I don't see {target} running - no matching port, tmux pane, or process.",
        answerable=True,
        escalate=False,
        kind="is_running",
    )


def _fmt_resources(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    res = snap.get("resources") or {}
    bits = []
    if res.get("load_avg"):
        bits.append("load " + "/".join(f"{x:.1f}" for x in res["load_avg"]))
    if res.get("mem_used_gb") is not None:
        bits.append(f"memory {res['mem_used_gb']} of {res['mem_total_gb']}GB used")
    if res.get("battery_pct") is not None:
        ac = "on AC" if res.get("ac_online") else "on battery"
        bits.append(f"battery {res['battery_pct']}% ({ac})")
    if res.get("disk_free_gb") is not None:
        bits.append(f"{res['disk_free_gb']}GB free on disk")
    if not bits:
        return StateAnswer(
            "I don't have resource numbers available right now.",
            answerable=False,
            escalate=True,
            kind="resources",
        )
    return StateAnswer(
        ", ".join(bits) + ".",
        answerable=True,
        escalate=False,
        kind="resources",
    )


def _fmt_unanswerable_failures(snap: dict, match: Optional[re.Match]) -> StateAnswer:
    return StateAnswer(
        "I don't have test or failure status in what I'm tracking right now - "
        "I only see running processes, ports, and git state, not build/test results.",
        answerable=False,
        escalate=True,
        kind="failures",
    )


_FORMATTERS: dict[str, Callable[[dict, Optional[re.Match]], StateAnswer]] = {
    "branch": _fmt_branch,
    "agent_doing": _fmt_agent_doing,
    "whats_running": _fmt_whats_running,
    "whats_happening": _fmt_whats_happening,
    "in_progress": _fmt_in_progress,
    "is_running": _fmt_is_running,
    "resources": _fmt_resources,
    "failures": _fmt_unanswerable_failures,
    "test_failures": _fmt_unanswerable_failures,
    "test_failures_count": _fmt_unanswerable_failures,
}


def _classify_kind(norm: str) -> tuple[Optional[str], Optional[re.Match]]:
    for kind, pattern in _PATTERNS:
        m = pattern.match(norm)
        if m:
            return kind, m
    return None, None


def answer(text: str, span: Optional[TurnSpan] = None) -> StateAnswer:
    """Answer a STATE_QUERY-classified `text` from live state.

    Marks `context_ready` once the snapshot is in hand and `task_complete`
    once the answer text is finalized. Never calls an LLM or touches the
    network beyond what `friday.state.snapshot()` already does.
    """
    norm = _normalize(text)
    kind, match = _classify_kind(norm)

    snap = state_mod.snapshot()
    if span is not None:
        span.mark("context_ready")

    if kind is None:
        result = StateAnswer(
            "I don't have a way to answer that from what I'm tracking right now.",
            answerable=False,
            escalate=True,
            kind=None,
        )
    else:
        result = _FORMATTERS[kind](snap, match)

    if span is not None:
        span.mark("task_complete")
    return result
