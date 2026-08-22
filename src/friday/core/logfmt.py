"""Colourise the event log on read, the way openclaw's gateway console does.

Why on read and not at write time: the daemon's stderr goes straight into
systemd's journal. ANSI escapes written there are stored verbatim, so every
`grep`, `journalctl --grep` and log-shipping tool downstream has to cope with
them forever. Writing plain text and colouring in the reader keeps the stored
form clean and still gives a live terminal the colour.

Format, matching openclaw:

    HH:MM:SS.mmm [subsystem] message key=value

  * timestamp   dim
  * [subsystem] one of six colours, picked by a stable hash of the name, so
                `wake` is always the same colour across runs and machines
  * message     coloured by level -- errors red, warnings yellow, else cyan
  * key=value   dim keys, plain values, so the numbers are what stands out
"""

from __future__ import annotations

import os
import re
import sys

RESET = "\033[0m"
DIM = "\033[2m"
_COLORS = {
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "red": "\033[31m",
}
_PALETTE = ("cyan", "green", "yellow", "blue", "magenta", "red")

# Subsystems whose meaning is fixed enough to be worth pinning, so a glance at
# colour alone distinguishes "she heard something" from "something went wrong".
_OVERRIDES = {"wake": "green", "reply": "cyan", "turn": "red", "mic": "magenta"}

_LINE = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3}) \[([a-z0-9-]+)\] ?(.*)$")
_FIELD = re.compile(r"(\b[a-z_]+)=(\S+)")


def _subsystem_color(name: str) -> str:
    if name in _OVERRIDES:
        return _COLORS[_OVERRIDES[name]]
    # Same hash openclaw uses (h*31 + char, truncated to int32), so a shared
    # subsystem name gets the same colour in both tools.
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return _COLORS[_PALETTE[abs(h) % len(_PALETTE)]]


def enabled(stream=None) -> bool:
    """Colour only for a real terminal, and never when NO_COLOR is set."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FRIDAY_COLOR") == "0":
        return False
    stream = stream or sys.stdout
    try:
        return stream.isatty()
    except Exception:  # noqa: BLE001
        return False


def colorize(line: str, *, color: bool = True) -> str:
    """Colour one event line. A line that is not an event is returned as-is.

    Untouched passthrough matters: the journal interleaves systemd's own
    "Started FRIDAY..." lines and any stray traceback with FRIDAY's events, and
    mangling or dropping those would hide exactly the output someone tailing
    the log is hunting for.
    """
    match = _LINE.match(line.rstrip("\n"))
    if not match or not color:
        return line.rstrip("\n")
    stamp, subsystem, rest = match.groups()

    level = _COLORS["cyan"]
    lowered = rest.lower()
    if subsystem == "turn" or "failed" in lowered or "error" in lowered:
        level = _COLORS["red"]
    elif "timed out" in lowered or "paused" in lowered or "preempted" in lowered:
        level = _COLORS["yellow"]

    def _field(m: "re.Match[str]") -> str:
        return f"{DIM}{m.group(1)}={RESET}{m.group(2)}"

    body = _FIELD.sub(_field, rest)
    # Re-open the level colour after each field, which reset it.
    body = body.replace(RESET, RESET + level)
    tint = _subsystem_color(subsystem)
    return (f"{DIM}{stamp}{RESET} {tint}[{subsystem}]{RESET} "
            f"{level}{body}{RESET}")


def stream_lines(source, out=None, *, color: bool = True) -> None:
    """Colourise `source` line by line onto `out`, flushing as it goes.

    Line-by-line and flushed because this is what `friday start` leaves running
    in the terminal: buffering would hold a wake event back until the next one
    arrived, which for a voice assistant is precisely the moment it matters.
    """
    out = out or sys.stdout
    for line in source:
        print(colorize(line, color=color), file=out, flush=True)
