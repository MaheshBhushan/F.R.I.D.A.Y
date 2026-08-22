"""Suite-wide hang watchdog.

A test that hangs is worse than a test that fails: it produces no output, and
under a CI runner or a `timeout` wrapper whose signal has been redirected it can
sit there indefinitely. One did exactly that here -- five hours on an unbounded
`subprocess.Popen.wait()` -- and nothing reported it.

SIGALRM rather than a plugin, because it needs no new dependency and it fires
even when the main thread is blocked in C (`waitpid`, `read`, PortAudio),
which is precisely where the hangs that matter happen. A thread-based timer
could not interrupt those.

The slowest legitimate test in this suite is under 10s, so the default is a very
loose 60s: this catches hangs, it does not police performance. Override with
FRIDAY_TEST_TIMEOUT (0 disables, e.g. when attaching a debugger).
"""

from __future__ import annotations

import os
import signal

import pytest

DEFAULT_TIMEOUT_S = 60


def _timeout() -> int:
    raw = os.environ.get("FRIDAY_TEST_TIMEOUT")
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_S


@pytest.fixture(autouse=True)
def _hang_watchdog(request):
    seconds = _timeout()
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise TimeoutError(
            f"{request.node.name} exceeded {seconds}s -- treated as a hang. "
            "Bound the wait, or raise FRIDAY_TEST_TIMEOUT if it is genuinely slow."
        )

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
