"""On-demand echo-canceller lifecycle.

The valuable cases are the ones that would silently cost something: unloading a
module we did not load (breaking someone else's audio), leaking a half-built
module across restarts, and turning a failed load into a dead assistant instead
of a degraded one.
"""

from __future__ import annotations

import asyncio

from friday.audio.echocancel import EchoCancelModule

HARDWARE_ONLY = """0\talsa_output...Speaker__sink.monitor\tPipeWire
1\talsa_input...Mic1__source\tPipeWire
"""
WITH_AEC = HARDWARE_ONLY + "2\techo-cancel-source\tPipeWire\n"


def _module(script):
    """`script` maps a pactl verb to (returncode, stdout). Records calls."""
    calls: list[tuple[str, ...]] = []

    async def pactl(*args: str):
        calls.append(args)
        return script(args, calls)

    return EchoCancelModule(pactl=pactl, appear_timeout=0.5), calls


def test_uses_an_existing_module_without_claiming_ownership():
    """A leftover config file or a second FRIDAY already loaded it. Unloading
    that on exit would break whatever else is using it."""
    def script(args, calls):
        if args[:2] == ("list", "short"):
            return 0, WITH_AEC
        raise AssertionError(f"must not have called {args}")

    m, calls = _module(script)
    status = asyncio.run(m.ensure_loaded())
    assert status.available and not status.owned
    assert "already present" in status.reason
    asyncio.run(m.release())
    assert not any(a[0] == "unload-module" for a in calls)


def test_loads_on_demand_and_unloads_on_release():
    state = {"loaded": False}

    def script(args, calls):
        if args[:2] == ("list", "short"):
            return 0, WITH_AEC if state["loaded"] else HARDWARE_ONLY
        if args[0] == "load-module":
            state["loaded"] = True
            return 0, "536870916\n"
        if args[0] == "unload-module":
            state["loaded"] = False
            return 0, ""
        raise AssertionError(args)

    m, calls = _module(script)
    status = asyncio.run(m.ensure_loaded())
    assert status.available and status.owned
    assert status.module_id == 536870916

    load = next(a for a in calls if a[0] == "load-module")
    # Pinning the masters is the point: an unpinned AEC follows the DEFAULT
    # sink and ends up cancelling against HDMI, where there is no acoustic loop.
    assert any(x.startswith("source_master=") and "Mic1" in x for x in load)
    assert any(x.startswith("sink_master=") and "Speaker" in x for x in load)
    assert any(x.startswith("aec_args=") for x in load)

    asyncio.run(m.release())
    assert ("unload-module", "536870916") in calls
    assert not state["loaded"]


def test_release_is_idempotent():
    def script(args, calls):
        if args[:2] == ("list", "short"):
            return 0, HARDWARE_ONLY
        if args[0] == "load-module":
            return 1, "Failure: module initialization failed"
        raise AssertionError(args)

    m, calls = _module(script)
    asyncio.run(m.ensure_loaded())
    asyncio.run(m.release())
    asyncio.run(m.release())
    assert not any(a[0] == "unload-module" for a in calls)


def test_failed_load_degrades_instead_of_raising():
    """No echo cancellation is far better than no assistant."""
    def script(args, calls):
        if args[:2] == ("list", "short"):
            return 0, HARDWARE_ONLY
        if args[0] == "load-module":
            return 1, "Failure: module-echo-cancel not found"
        raise AssertionError(args)

    m, _ = _module(script)
    status = asyncio.run(m.ensure_loaded())
    assert not status.available and not status.owned
    assert "load-module failed" in status.reason


def test_missing_pactl_degrades_instead_of_raising():
    async def pactl(*args):
        raise FileNotFoundError("pactl")

    m = EchoCancelModule(pactl=pactl, appear_timeout=0.1)
    status = asyncio.run(m.ensure_loaded())
    assert not status.available
    assert "FileNotFoundError" in status.reason


def test_module_that_never_appears_is_not_leaked():
    """Loaded but the node never showed up: unload it, or every restart leaks
    another half-built module into the graph."""
    def script(args, calls):
        if args[:2] == ("list", "short"):
            return 0, HARDWARE_ONLY          # never appears
        if args[0] == "load-module":
            return 0, "42\n"
        if args[0] == "unload-module":
            return 0, ""
        raise AssertionError(args)

    m, calls = _module(script)
    status = asyncio.run(m.ensure_loaded())
    assert not status.available
    assert "never appeared" in status.reason
    assert ("unload-module", "42") in calls
