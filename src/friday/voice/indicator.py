"""Visible "she's talking" indicator.

FRIDAY has no GUI, so the indicator is two cheap things rather than a tray
applet: an in-place ANSI status line on stderr, and a one-word state file at
`~/.friday/status` that any external widget (waybar, a Plasma command
plasmoid, a tmux status segment) can poll without knowing anything about this
process.

Deliberately not a Qt/GTK tray icon: that needs its own event loop inside a
process whose whole design is one asyncio loop, and it would be invisible on
a headless or SSH-attached run. A state file is readable from everywhere,
including the tests.

Thread-safety matters here for one specific reason: playback state changes
sit next to PortAudio callbacks, which run on the audio thread. Nothing in
this module may be called FROM such a callback -- a file write there can
stall the buffer and cause a dropout. Callers transition around the stream
(before open / after close), never inside it. The lock only guards against
two asyncio tasks interleaving a redraw.

Set FRIDAY_INDICATOR=0 to disable output entirely (state tracking still
works, so tests and callers don't need to care).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Callable, IO, Iterator, Optional

STATUS_PATH = Path.home() / ".friday" / "status"


class State(Enum):
    """What FRIDAY is doing, as far as the user can perceive it."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    TALKING = "talking"
    SUSPENDED = "suspended"


# glyph, ANSI colour, label. The glyphs are filled/hollow rather than
# colour-only so the state is still distinguishable on a mono terminal.
_LOOK: dict[State, tuple[str, str, str]] = {
    State.IDLE: ("○", "2", "idle"),          # hollow circle, dim
    State.LISTENING: ("◉", "36", "listening"),  # fisheye, cyan
    State.THINKING: ("◌", "33", "thinking"),    # dotted circle, yellow
    State.TALKING: ("◆", "32", "talking"),      # filled diamond, green
    State.SUSPENDED: ("⊘", "35", "mic paused"),  # slashed circle, magenta
}

_lock = threading.Lock()
_state = State.IDLE

# Observers are notified on every transition, whether or not the terminal
# indicator is enabled: FRIDAY_INDICATOR=0 means "do not draw", not "do not
# report". The gateway relies on this to push state.changed to its clients in
# a headless service where there is no tty at all.
_observers: "list[Callable[[State, str], None]]" = []


def subscribe(callback: "Callable[[State, str], None]") -> None:
    """Register `callback(state, detail)`, invoked on every transition.

    Callbacks run on whichever thread called set_state, outside the module
    lock, and their exceptions are swallowed -- an observer must never be able
    to break a turn. Anything that needs to touch an event loop should hand
    off with call_soon_threadsafe rather than doing work here.
    """
    with _lock:
        if callback not in _observers:
            _observers.append(callback)


def unsubscribe(callback: "Callable[[State, str], None]") -> None:
    with _lock:
        with contextlib.suppress(ValueError):
            _observers.remove(callback)


def _enabled() -> bool:
    return os.environ.get("FRIDAY_INDICATOR", "1") != "0"


def current() -> State:
    """The last state set. Cheap; no I/O."""
    return _state


def _render(state: State, stream: IO[str], detail: str = "") -> None:
    glyph, colour, label = _LOOK[state]
    if detail:
        label = f"{label} · {detail}"
    # \r + erase-to-end-of-line keeps this to a single reused terminal line
    # instead of scrolling a log of every transition.
    stream.write(f"\r\x1b[2K\x1b[{colour}m{glyph}\x1b[0m friday: {label}")
    stream.flush()


def _publish(state: State) -> None:
    """Write the state word atomically so a reader never sees a partial line."""
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(state.value + "\n")
        os.replace(tmp, STATUS_PATH)
    except OSError:
        # An indicator must never be able to take down a turn.
        pass


def set_state(state: State, *, detail: str = "",
              stream: Optional[IO[str]] = None) -> None:
    """Transition to `state`, redrawing the line and updating the state file.

    `detail` is appended after a separator, for states that mean nothing on
    their own -- "mic paused" needs to say who took it.

    Never call this from an audio callback -- see the module docstring.
    """
    global _state
    with _lock:
        _state = state
        # FRIDAY_INDICATOR=0 suppresses the *rendering*, not the transition:
        # `current()` and the observers stay truthful either way, so a headless
        # gateway still reports state with the terminal line switched off.
        if _enabled():
            out = stream if stream is not None else sys.stderr
            if out.isatty():
                _render(state, out, detail)
            _publish(state)
    _notify(state, detail)


def _notify(state: State, detail: str) -> None:
    """Fan out to observers with the lock released.

    Holding _lock across a callback would deadlock the moment an observer
    called back into subscribe/unsubscribe -- which the gateway does when a
    client disconnects mid-transition.
    """
    with _lock:
        observers = list(_observers)
    for callback in observers:
        try:
            callback(state, detail)
        except Exception:  # noqa: BLE001 - an observer cannot break a turn
            pass


@contextmanager
def during(state: State, *, back_to: State = State.IDLE,
           stream: Optional[IO[str]] = None) -> Iterator[None]:
    """Hold `state` for the block, then return to `back_to` even on error.

    The `finally` is the point: a synthesis failure or a hard preempt mid
    utterance must not leave the indicator stuck reading "talking".
    """
    set_state(state, stream=stream)
    try:
        yield
    finally:
        set_state(back_to, stream=stream)


def settle(*, stream: Optional[IO[str]] = None) -> None:
    """Return to IDLE, but only if still THINKING.

    A reasoning turn's token stream is consumed by the speaker, so by the time
    the turn ends the state is usually TALKING with audio still playing --
    forcing IDLE there would blank the indicator mid sentence. If nothing ever
    spoke (a text-only turn, a tool-only turn) the state is still THINKING and
    this is what stops it sticking there.
    """
    if current() is State.THINKING:
        set_state(State.IDLE, stream=stream)


def clear(*, stream: Optional[IO[str]] = None) -> None:
    """Erase the status line, e.g. at shutdown."""
    if not _enabled():
        return
    out = stream if stream is not None else sys.stderr
    if out.isatty():
        out.write("\r\x1b[2K")
        out.flush()
