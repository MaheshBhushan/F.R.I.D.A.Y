"""Proactive interrupts: FRIDAY notices something and speaks first.

The pipeline is deliberately one-way and has exactly one producer
(`watch`, a polling loop over `state.snapshot()` / `agents.poll()` diffs)
and one consumer (`Attention.run`, draining an `asyncio.Queue`):

    event -> score() -> gate() -> enrich() -> interrupt()

`score()` is the whole point of this module and it is *rule-only*: a
`dict[event_type -> int]` table plus a handful of small functions that
bump the score using fields already present in the snapshot. No LLM, no
network, no I/O -- scoring costs microseconds and can be called on every
poll tick without a budget. The language model is used only by
`interrupt()`, to *word* a sentence about a decision the rules already
made; by then the decision to speak is final.

Between scoring and speech sits `enrich()`: a bounded, read-only look at
the log/state field that explains the cause, so the interrupt says "the
dev server died, the log tail points at PostgreSQL" instead of just
"something died". It is wrapped in `asyncio.wait_for` -- a slow log read
degrades the sentence, it never delays the interrupt past
`ENRICH_TIMEOUT_S`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from friday.brain import complete
from friday.core.spans import TurnSpan, start_turn

# --- the event --------------------------------------------------------------


@dataclass
class Event:
    """One observed change worth scoring.

    `subject` is what the event is about (a process name, a tmux session)
    and doubles as the debounce key alongside `type`. `data` carries the
    raw fields the rule functions read -- never anything that needs
    fetching, only what the watcher already had in hand.
    """

    type: str
    subject: str = ""
    detail: str = ""
    data: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    @property
    def key(self) -> tuple[str, str]:
        return (self.type, self.subject)


# --- scoring: rules only, never an LLM --------------------------------------

# Base importance per event type. Anything absent scores 0 and is ignored.
SCORES: dict[str, int] = {
    # worth interrupting for
    "dev_server_died": 80,
    "agent_needs_permission": 90,
    "agent_failed": 70,
    "build_failed": 65,
    "tests_failed": 60,
    "disk_low": 55,
    "battery_low": 30,  # only matters off AC; see _bump_battery
    "agent_idle": 25,  # only matters if it has been idle a while
    # noise
    "agent_running": 5,
    "file_changed": 5,
    "log_line": 1,
    "git_branch_changed": 10,
    "git_dirty_changed": 5,
    "port_opened": 10,
    "process_started": 5,
    "load_changed": 2,
}

THRESHOLD = 50


def _bump_battery(event: Event) -> int:
    """Battery level only matters when nothing is charging it."""
    if event.type != "battery_low":
        return 0
    if event.data.get("ac_online"):
        return -30
    pct = event.data.get("battery_pct")
    return 40 if pct is not None and pct <= 10 else 25


def _bump_disk(event: Event) -> int:
    """Under 2GB free is about to break a build, not just a warning."""
    if event.type != "disk_low":
        return 0
    free = event.data.get("disk_free_gb")
    return 30 if free is not None and free < 2 else 0


def _bump_stalled_agent(event: Event) -> int:
    """An agent idle for a long time has probably finished or wedged."""
    if event.type != "agent_idle":
        return 0
    return 30 if event.data.get("since_change_secs", 0) >= 120 else 0


def _bump_named_dev_server(event: Event) -> int:
    """A death we can name is more useful to report than an anonymous one."""
    if event.type != "dev_server_died":
        return 0
    return 5 if event.subject else -20


RULES: tuple[Callable[[Event], int], ...] = (
    _bump_battery,
    _bump_disk,
    _bump_stalled_agent,
    _bump_named_dev_server,
)


def score(event: Event) -> int:
    """Rule-only importance for `event`. Pure arithmetic over dicts: no
    LLM, no network, no subprocess, no disk. Microseconds."""
    total = SCORES.get(event.type, 0)
    for rule in RULES:
        total += rule(event)
    return total


# --- nag control ------------------------------------------------------------

# Policy: an identical (type, subject) may interrupt at most once per
# COOLDOWN_S; any two interrupts are at least MIN_GAP_S apart; and no more
# than MAX_PER_WINDOW interrupts happen in any WINDOW_S span.
COOLDOWN_S = 300.0
MIN_GAP_S = 30.0
MAX_PER_WINDOW = 3
WINDOW_S = 600.0

ENRICH_TIMEOUT_S = 1.0


# --- enrichment -------------------------------------------------------------


async def _default_investigate(event: Event) -> str:
    """Read-only look at whatever the watcher pointed us at: the log tail
    it captured, or the state field that explains the cause."""
    tail = event.data.get("log_tail")
    if isinstance(tail, (list, tuple)):
        tail = "\n".join(tail)
    if tail:
        return str(tail).strip()
    log_path = event.data.get("log_path")
    if log_path:
        with open(log_path, "r", errors="replace") as f:
            return "\n".join(f.read().splitlines()[-8:]).strip()
    return str(event.data.get("cause", "")).strip()


async def enrich(
    event: Event,
    *,
    investigate: Callable[[Event], Awaitable[str]] = _default_investigate,
    timeout: float = ENRICH_TIMEOUT_S,
) -> str:
    """Gather evidence about `event`, giving up after `timeout` seconds.

    Returns "" on timeout or failure: a slow log read costs us detail, it
    never stalls the interrupt.
    """
    try:
        return await asyncio.wait_for(investigate(event), timeout)
    except (asyncio.TimeoutError, OSError):
        return ""


def compose_prompt(event: Event, evidence: str) -> str:
    """The one place the LLM is involved: wording a decision already made."""
    lines = [
        "You are interrupting the user unprompted because this just happened.",
        f"Event: {event.type} ({event.subject or 'unknown subject'}).",
    ]
    if event.detail:
        lines.append(f"Detail: {event.detail}")
    if evidence:
        lines.append(f"Evidence gathered: {evidence}")
    else:
        lines.append("Evidence gathered: none (investigation timed out).")
    lines.append(
        "In one or two spoken sentences, name what happened, say what the "
        "evidence points at, and offer to investigate."
    )
    return "\n".join(lines)


# --- the attention layer ----------------------------------------------------


async def _play_ack_default(name: str, span: Optional[TurnSpan]) -> None:
    from friday.voice.ack import play_ack

    await asyncio.to_thread(play_ack, name, span)


class Attention:
    """Scores incoming events and, for the few that pass, preempts speech
    with a cached lead-in plus a streamed detail sentence."""

    def __init__(
        self,
        speaker: Any,
        transport: Any,
        *,
        ack_name: str = "sir",
        play_ack: Callable[[str, Optional[TurnSpan]], Awaitable[None]] = _play_ack_default,
        investigate: Callable[[Event], Awaitable[str]] = _default_investigate,
        threshold: int = THRESHOLD,
        enrich_timeout: float = ENRICH_TIMEOUT_S,
        memory: Any = None,
        spans_path: Optional[Any] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._speaker = speaker
        self._transport = transport
        self._ack_name = ack_name
        self._play_ack = play_ack
        self._investigate = investigate
        self._threshold = threshold
        self._enrich_timeout = enrich_timeout
        self._memory = memory
        self._spans_path = spans_path
        self._now = now
        self._last_per_key: dict[tuple[str, str], float] = {}
        self._recent: list[float] = []
        self.interrupts: list[Event] = []
        self.suppressed: list[tuple[Event, str]] = []

    # -- gating --------------------------------------------------------

    def gate(self, event: Event) -> tuple[bool, str]:
        """Whether `event` may interrupt right now, and why not if not."""
        points = score(event)
        if points < self._threshold:
            return False, f"below_threshold:{points}"
        now = self._now()
        last = self._last_per_key.get(event.key)
        if last is not None and (now - last) < COOLDOWN_S:
            return False, "cooldown"
        self._recent = [t for t in self._recent if (now - t) < WINDOW_S]
        if self._recent and (now - self._recent[-1]) < MIN_GAP_S:
            return False, "min_gap"
        if len(self._recent) >= MAX_PER_WINDOW:
            return False, "rate_limit"
        return True, f"pass:{points}"

    def _commit(self, event: Event) -> None:
        now = self._now()
        self._last_per_key[event.key] = now
        self._recent.append(now)

    # -- the emit path -------------------------------------------------

    async def interrupt(self, event: Event, evidence: str, span: TurnSpan) -> str:
        """Preempt in-flight speech, play the cached lead-in, then stream
        the detail sentence. Returns the text that was spoken."""
        if getattr(self._speaker, "is_speaking", False):
            self._speaker.stop()
        await self._play_ack(self._ack_name, span)
        prompt = compose_prompt(event, evidence)
        spoken: list[str] = []

        async def _tokens():
            async for token in complete(prompt, self._transport, span=span):
                spoken.append(token)
                yield token

        await self._speaker.speak(_tokens(), span=span)
        return "".join(spoken)

    async def handle(self, event: Event) -> Optional[str]:
        """Score, gate, enrich, speak. Returns the spoken text, or None if
        the event was not worth interrupting for."""
        allowed, reason = self.gate(event)
        if not allowed:
            self.suppressed.append((event, reason))
            return None
        self._commit(event)
        self.interrupts.append(event)
        span = (
            start_turn("proactive")
            if self._spans_path is None
            else start_turn("proactive", path=self._spans_path)
        )
        evidence = await enrich(
            event, investigate=self._investigate, timeout=self._enrich_timeout
        )
        span.mark("context_ready")
        text = await self.interrupt(event, evidence, span)
        span.write()
        if self._memory is not None:
            self._memory.record(
                f"proactive interrupt: {event.type} ({event.subject}) -> {text}",
                importance=1.0,
            )
        return text

    async def run(self, queue: "asyncio.Queue[Optional[Event]]") -> None:
        """Drain `queue` until a `None` sentinel arrives."""
        while True:
            event = await queue.get()
            if event is None:
                return
            await self.handle(event)


# --- the watcher: one loop over snapshot/poll diffs -------------------------


def diff_events(
    prev: Optional[dict],
    snap: dict,
    prev_agents: Optional[dict] = None,
    agents: Optional[dict] = None,
) -> list[Event]:
    """Turn the change between two observations into events. `agents` maps
    session name -> the dict `agents.poll()` returned."""
    events: list[Event] = []
    agents = agents or {}
    prev_agents = prev_agents or {}

    if prev is not None:
        prev_ports = {p["port"]: p for p in prev.get("listening_ports") or []}
        for port, info in prev_ports.items():
            if port not in {p["port"] for p in snap.get("listening_ports") or []}:
                events.append(
                    Event(
                        type="dev_server_died",
                        subject=info.get("process") or f"port {port}",
                        detail=f"stopped listening on port {port}",
                        data={"port": port, "was": info},
                    )
                )
        for p in snap.get("listening_ports") or []:
            if p["port"] not in prev_ports:
                events.append(
                    Event(type="port_opened", subject=str(p["port"]), data={"port": p["port"]})
                )
        prev_git = prev.get("git") or {}
        git = snap.get("git") or {}
        if git.get("branch") and prev_git.get("branch") != git.get("branch"):
            events.append(
                Event(type="git_branch_changed", subject=git["branch"], data=dict(git))
            )

    res = snap.get("resources") or {}
    free = res.get("disk_free_gb")
    if free is not None and free < 5:
        events.append(Event(type="disk_low", subject="disk", data=dict(res)))
    pct = res.get("battery_pct")
    if pct is not None and pct <= 20:
        events.append(Event(type="battery_low", subject="battery", data=dict(res)))

    for session, info in agents.items():
        status = info.get("status")
        if status == "permission":
            events.append(
                Event(
                    type="agent_needs_permission",
                    subject=session,
                    detail="is waiting on an approval",
                    data=dict(info),
                )
            )
        elif status == "idle" and (prev_agents.get(session) or {}).get("status") != "idle":
            events.append(Event(type="agent_idle", subject=session, data=dict(info)))
        elif status == "running":
            events.append(Event(type="agent_running", subject=session, data=dict(info)))

    return events


async def watch(
    queue: "asyncio.Queue[Optional[Event]]",
    *,
    snapshot: Callable[[], dict],
    poll_agents: Callable[[], dict] = lambda: {},
    interval: float = 2.0,
    ticks: Optional[int] = None,
) -> None:
    """Poll state, diff it against the previous observation, and put the
    resulting events on `queue`. Stops after `ticks` polls if given."""
    prev: Optional[dict] = None
    prev_agents: dict = {}
    count = 0
    while ticks is None or count < ticks:
        snap = await asyncio.to_thread(snapshot)
        agents = await asyncio.to_thread(poll_agents)
        for event in diff_events(prev, snap, prev_agents, agents):
            await queue.put(event)
        prev, prev_agents = snap, agents
        count += 1
        if ticks is None or count < ticks:
            await asyncio.sleep(interval)
