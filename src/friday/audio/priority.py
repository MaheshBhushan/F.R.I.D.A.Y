"""Who currently owns the microphone, and how important they are.

Five tiers, P0 highest and FRIDAY last:

    P0  emergency / system capture
    P1  calls and meetings
    P2  explicit recording apps
    P3  other interactive apps      <- unknown applications land here
    P4  FRIDAY

Priorities are numerically ASCENDING with decreasing importance, so "preempts
FRIDAY" is simply `priority < Priority.P4_FRIDAY`. Anything unrecognised gets
P3, which still preempts: FRIDAY is least-privileged by design, so an
unidentified recorder wins rather than being fought.

Detection uses `pactl list source-outputs`, which ships with PipeWire's Pulse
shim -- no new dependency. Two entries must be filtered out or FRIDAY is
preempted forever:

  * **Virtual nodes.** `module-echo-cancel` is itself a source-output
    (`node.name = echo-cancel-capture`, `node.virtual = true`, `Client: n/a`).
    It is infrastructure and permanently present; counting it means FRIDAY is
    preempted by her own echo canceller for eternity.
  * **FRIDAY's own stream**, matched on `application.process.id` against this
    process's PID. Exact, unlike the client name: PortAudio's Pulse client
    announces itself as "ALSA plug-in [python3.11]", which is neither unique
    nor stable.

Override the table with a JSON object in `~/.friday/mic-priority.json`, e.g.
`{"obs": 1}` to treat OBS as a call, or `{"some-app": 4}` to let FRIDAY keep
listening alongside it (equal priority does not preempt).
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

PRIORITY_PATH = Path.home() / ".friday" / "mic-priority.json"


class Priority(IntEnum):
    """Lower value == higher priority. FRIDAY is last on purpose."""

    P0_SYSTEM = 0        # emergency / system capture
    P1_CALLS = 1         # calls and meetings
    P2_RECORDING = 2     # explicit recording and streaming apps
    P3_INTERACTIVE = 3   # other interactive apps; the default for unknowns
    P4_FRIDAY = 4        # FRIDAY


FRIDAY_PRIORITY = Priority.P4_FRIDAY
DEFAULT_PRIORITY = Priority.P3_INTERACTIVE

# Matched as lowercase substrings against application.process.binary and
# application.name. The strongest (numerically lowest) match wins.
PRIORITY: dict[str, int] = {
    # P1 -- calls and meetings. Browsers live here rather than P3 because a
    # browser holding the microphone is, in practice, always a call.
    "zoom": Priority.P1_CALLS,
    "teams": Priority.P1_CALLS,
    "webex": Priority.P1_CALLS,
    "skype": Priority.P1_CALLS,
    "discord": Priority.P1_CALLS,
    "slack": Priority.P1_CALLS,
    "element": Priority.P1_CALLS,
    "signal": Priority.P1_CALLS,
    "telegram": Priority.P1_CALLS,
    "jitsi": Priority.P1_CALLS,
    "mumble": Priority.P1_CALLS,
    "firefox": Priority.P1_CALLS,
    "chromium": Priority.P1_CALLS,
    "chrome": Priority.P1_CALLS,
    "brave": Priority.P1_CALLS,
    "vivaldi": Priority.P1_CALLS,
    "epiphany": Priority.P1_CALLS,
    # P2 -- explicit recording. Interrupting these ruins a take.
    "obs": Priority.P2_RECORDING,
    "audacity": Priority.P2_RECORDING,
    "ardour": Priority.P2_RECORDING,
    "reaper": Priority.P2_RECORDING,
    "ffmpeg": Priority.P2_RECORDING,
    "arecord": Priority.P2_RECORDING,
    "parecord": Priority.P2_RECORDING,
    "pacat": Priority.P2_RECORDING,
    "sox": Priority.P2_RECORDING,
    # Push-to-talk dictation. A take here is as interruption-sensitive as a
    # recording: FRIDAY talking over a half-dictated sentence corrupts the
    # text being typed, so this sits at P2 rather than the P3 default.
    "voicewin": Priority.P2_RECORDING,
    "nerd-dictation": Priority.P2_RECORDING,
    # P4 -- listed so the CLI can show FRIDAY's own tier.
    "friday": Priority.P4_FRIDAY,
}

_SOURCE_OUTPUT_RE = re.compile(r"^Source Output #(\d+)")
_CLIENT_RE = re.compile(r"^\s*Client:\s*(.+)$")
_PROP_RE = re.compile(r'^\s*([\w.]+) = "(.*)"$')


@dataclass(frozen=True)
class Owner:
    """One non-virtual application currently capturing audio."""

    index: int
    name: str
    binary: str
    pid: Optional[int]
    priority: int

    @property
    def label(self) -> str:
        return self.binary or self.name or f"#{self.index}"

    @property
    def preempts_friday(self) -> bool:
        return self.priority < FRIDAY_PRIORITY

    @property
    def tier(self) -> str:
        with contextlib.suppress(ValueError):
            return Priority(self.priority).name
        return f"P?({self.priority})"


def load_priorities() -> dict[str, int]:
    """Built-in table with `~/.friday/mic-priority.json` merged over it."""
    table = dict(PRIORITY)
    try:
        override = json.loads(PRIORITY_PATH.read_text())
    except (OSError, ValueError):
        return table
    if isinstance(override, dict):
        for key, value in override.items():
            with contextlib.suppress(TypeError, ValueError):
                table[str(key).lower()] = int(value)
    return table


def priority_of(name: str, binary: str,
                table: Optional[dict[str, int]] = None) -> int:
    """Strongest (numerically lowest) priority matching either identifier."""
    table = table if table is not None else load_priorities()
    haystack = f"{name} {binary}".lower()
    matches = [p for key, p in table.items() if key in haystack]
    return min(matches) if matches else int(DEFAULT_PRIORITY)


def parse_owners(text: str, *, own_pid: int,
                 table: Optional[dict[str, int]] = None) -> list[Owner]:
    """Parse `pactl list source-outputs`, dropping virtual nodes and our own."""
    table = table if table is not None else load_priorities()
    out: list[Owner] = []
    index: Optional[int] = None
    client = ""
    props: dict[str, str] = {}

    def _flush() -> None:
        if index is None:
            return
        if props.get("node.virtual") == "true" or client in ("", "n/a"):
            return
        pid_raw = props.get("application.process.id", "")
        pid = int(pid_raw) if pid_raw.isdigit() else None
        if pid == own_pid:
            return
        name = props.get("application.name", props.get("node.name", ""))
        binary = props.get("application.process.binary", "")
        out.append(Owner(index, name, binary, pid,
                         priority_of(name, binary, table)))

    for line in text.splitlines():
        head = _SOURCE_OUTPUT_RE.match(line)
        if head:
            _flush()
            index, client, props = int(head.group(1)), "", {}
            continue
        if index is None:
            continue
        found = _CLIENT_RE.match(line)
        if found:
            client = found.group(1).strip()
            continue
        prop = _PROP_RE.match(line)
        if prop:
            props[prop.group(1)] = prop.group(2)
    _flush()
    return out
