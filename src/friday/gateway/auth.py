"""Bearer-token auth for the gateway, with a brute-force brake.

The gateway can speak, listen, run tools and read memory, so an unauthenticated
port here is worth strictly more to an attacker than a shell -- it is a shell
with a microphone. Three properties matter, in this order:

1.  No token, no service. There is no anonymous mode, not even on loopback.
    A default-open localhost port is reachable from every process on the box,
    including a browser tab via DNS rebinding.
2.  Constant-time comparison. `==` on secrets leaks their prefix through
    timing, and this endpoint is designed to be hammered.
3.  A per-peer failure brake, so guessing is slow even with the above.

The token lives in a 0600 file rather than an environment variable or the
config: env vars leak through `/proc/<pid>/environ` and into crash reports,
and FRIDAY's own `read_file` tool already refuses that path specifically to
stop the model reading its own credentials back out.
"""

from __future__ import annotations

import os
import secrets
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from friday.gateway.protocol import ErrorCode, ProtocolError

TOKEN_PATH = Path(os.environ.get("FRIDAY_GATEWAY_TOKEN_FILE",
                                 Path.home() / ".friday" / "gateway-token"))
TOKEN_ENV = "FRIDAY_GATEWAY_TOKEN"
TOKEN_BYTES = 32

# A human retyping a token gets a handful of attempts; a script gets throttled
# into uselessness. 5 failures per peer per minute makes an online guess against
# 256 bits of entropy take longer than the heat death of the machine.
MAX_FAILURES = 5
WINDOW_SECONDS = 60.0


def load_or_create_token(path: Optional[Path] = None) -> str:
    """Return the gateway token, minting one on first run.

    The env var wins when set so a test or a container can inject a known
    value without touching the user's real token file.

    Creation is atomic-ish by way of mode-on-open: the file is opened with
    0600 already applied rather than written and then chmod-ed, because the
    window between those two steps is exactly when another user can open it.
    """
    injected = os.environ.get(TOKEN_ENV)
    if injected:
        return injected

    target = path or TOKEN_PATH
    if target.exists():
        token = target.read_text().strip()
        if token:
            _warn_if_world_readable(target)
            return token
        # An empty token file is worse than a missing one: it reads as
        # "configured" while authenticating nobody. Replace it.

    target.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")
    return token


def _warn_if_world_readable(target: Path) -> None:
    """Complain about a loose token file, but keep running.

    Refusing to start would be defensible, but this is a voice assistant the
    user talks to; bricking it over a permission bit trains them to work
    around the check. Say it loudly instead.
    """
    try:
        mode = target.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(f"[friday] warning: {target} is readable beyond its owner; "
              f"run: chmod 600 {target}")


@dataclass
class RateLimiter:
    """Sliding-window failure counter, keyed by peer.

    Counts *failures only*. Rate-limiting successful traffic would throttle
    the legitimate UI, which polls; the thing worth limiting is wrong guesses.
    """

    max_failures: int = MAX_FAILURES
    window_seconds: float = WINDOW_SECONDS
    # Injectable clock so the tests can prove window expiry without sleeping.
    now: "object" = time.monotonic
    _failures: dict[str, list[float]] = field(default_factory=dict)

    def _prune(self, peer: str, at: float) -> list[float]:
        cutoff = at - self.window_seconds
        kept = [t for t in self._failures.get(peer, []) if t > cutoff]
        if kept:
            self._failures[peer] = kept
        else:
            self._failures.pop(peer, None)
        return kept

    def check(self, peer: str) -> None:
        """Raise if `peer` has burned its budget."""
        at = self.now()  # type: ignore[operator]
        recent = self._prune(peer, at)
        if len(recent) >= self.max_failures:
            retry_in = self.window_seconds - (at - recent[0])
            raise ProtocolError(
                ErrorCode.RATE_LIMITED,
                f"too many failed attempts; retry in {max(retry_in, 0.0):.0f}s",
            )

    def record_failure(self, peer: str) -> None:
        at = self.now()  # type: ignore[operator]
        self._prune(peer, at)
        self._failures.setdefault(peer, []).append(at)

    def reset(self, peer: str) -> None:
        """Clear a peer's history after a success, so one typo is not sticky."""
        self._failures.pop(peer, None)


def verify(presented: object, expected: str) -> None:
    """Constant-time token check. Raises UNAUTHORIZED on any mismatch.

    A non-string `presented` is treated as a failed attempt rather than a
    malformed request: a client sending `{"token": null}` is guessing, and
    reporting it as `bad_request` would let a prober distinguish "wrong shape"
    from "wrong value" and skip straight to the interesting case.
    """
    if not isinstance(presented, str) or not presented:
        raise ProtocolError(ErrorCode.UNAUTHORIZED, "missing or invalid token")
    if not secrets.compare_digest(presented, expected):
        raise ProtocolError(ErrorCode.UNAUTHORIZED, "invalid token")
