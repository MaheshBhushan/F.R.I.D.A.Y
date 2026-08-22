"""One line per thing that happens, so `friday logs -f` is worth watching.

The daemon used to emit three lines per start and then nothing at all: mic
pause, mic resume, and errors. That is enough to know the process is alive and
nothing whatsoever about what it is doing, which made every voice-path bug a
guessing game -- there was no way to tell "the wake word never fired" from
"it fired and STT returned empty" from "the reply was empty".

Format is deliberately flat text, not JSON: the primary consumer is a human
reading `journalctl -f`, and spans.jsonl already exists for machine analysis.

    22:41:07.412 [wake]  score=0.81 model=alexa preroll=1.50s
    22:41:08.903 [stt]   "what branch am i on" ms=1491
    22:41:08.906 [route] tier=state_query
    22:41:09.101 [reply] "You are on feat/session-overlay-ids (dirty)." ms=195

The `[subsystem]` bracket form is openclaw's gateway console format, so the
two are readable side by side and one colouriser handles both. Colour is NOT
applied here: this stream goes to the journal, where escape codes would be
stored verbatim and break `grep`. `friday logs` colours on read instead.

Everything goes to stderr, which is where systemd's journal and the
direct-mode log file both already collect from -- so this needs no new
plumbing and cannot fight with the indicator's use of stdout.

Levels: the default (`info`) covers the turn lifecycle. `debug` adds the
high-rate stuff -- per-frame wake scores, STT interims -- that would otherwise
bury the signal. Set FRIDAY_LOG=debug to get it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Optional

_LEVELS = {"silent": 0, "info": 1, "debug": 2}
_ENV = "FRIDAY_LOG"

# Serialises writes. Turn work runs as a task on the loop while the wake
# detector's scores come through the same loop, but TTS and the audio manager
# touch threads -- two interleaved writes produce a spliced, unreadable line.
_lock = threading.Lock()
_stream = None


def _level() -> int:
    return _LEVELS.get(os.environ.get(_ENV, "info").strip().lower(), 1)


def set_stream(stream) -> None:
    """Redirect output. Tests use this; the daemon leaves it on stderr."""
    global _stream
    _stream = stream


def _out():
    return _stream if _stream is not None else sys.stderr


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def emit(kind: str, message: str = "", *, level: str = "info", **fields: Any) -> None:
    """Write one event. Never raises: logging cannot be allowed to end a turn."""
    if _LEVELS.get(level, 1) > _level():
        return
    try:
        # Local time to the millisecond. Wall-clock, not monotonic: this is for
        # correlating against what the user remembers saying and when.
        now = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
        parts = [f"{stamp} [{kind}]"]
        if message:
            parts.append(message)
        parts.extend(f"{k}={_fmt(v)}" for k, v in fields.items() if v is not None)
        out = _out()
        with _lock:
            print(" ".join(parts), file=out, flush=True)
    except Exception:  # noqa: BLE001 - a broken log must never break the loop
        pass


def debug(kind: str, message: str = "", **fields: Any) -> None:
    emit(kind, message, level="debug", **fields)


def quote(text: Optional[str], limit: int = 120) -> str:
    """Render a transcript/reply on one line, truncated, so a multi-line
    reply cannot forge extra log entries."""
    if not text:
        return '""'
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return f'"{flat}"'
