"""Single-process asyncio skeleton for FRIDAY.

Runs independent long-lived loops (voice, agent) under a supervisor so
one blocking/erroring loop cannot stall the others, and shuts down
cleanly on SIGINT/SIGTERM. `--selftest` drives synthetic turns through
every span stage instead of starting the real loops.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import random
import signal

from friday.core import events
from friday.core.spans import STAGES, start_turn

REFLEX_STAGES = (
    "speech_started",
    "speech_ended_vad",
    "stt_final",
    "intent_classified",
    "ack_audible",
    "task_complete",
)


# Published so the gateway can report on the voice loop without owning it.
# A module-level slot rather than a constructor argument because the two loops
# are started as independent siblings: neither may block on the other existing,
# and the gateway must come up even when the voice loop never does.
_LIVE_LOOP: "object | None" = None


def live_loop() -> "object | None":
    """The running VoiceLoop, or None if it never started or has stopped."""
    return _LIVE_LOOP


async def voice_loop(stop: asyncio.Event) -> None:
    """Run the assembled wake -> STT -> route -> ack -> brain -> TTS loop.

    Missing credentials are reported and then waited out rather than raised:
    the supervisor's job is to keep the other loops alive, and exiting the
    whole process because one key is unexported is not that.
    """
    global _LIVE_LOOP
    from friday import loop as voice

    try:
        lp = voice.build_live()
    except SystemExit as exc:
        events.emit("voice", "loop not started", error=str(exc))
        events.emit("voice", "gateway stays up; health will report voice_loop=false")
        await stop.wait()
        return
    _LIVE_LOOP = lp
    try:
        await lp.run(stop)
    finally:
        # Cleared on the way out so `health` reports the truth during shutdown
        # instead of advertising a loop that is no longer pumping audio.
        _LIVE_LOOP = None


async def gateway_loop(stop: asyncio.Event) -> None:
    """Serve the WebSocket control plane alongside the voice loop.

    A bind failure is fatal to this loop but not to the process: the usual
    cause is a second FRIDAY already running, and in that case the right
    outcome is a loud message plus a still-working voice loop, not a dead
    assistant. FRIDAY_GATEWAY=0 skips it entirely.
    """
    if os.environ.get("FRIDAY_GATEWAY", "1") == "0":
        await stop.wait()
        return

    from friday.gateway.server import Gateway

    gateway = Gateway(loop_ref=live_loop)
    try:
        await gateway.serve_forever(stop)
    except OSError as exc:
        events.emit("gateway", "could not bind",
                    addr=f"{gateway.host}:{gateway.port}", error=str(exc))
        events.emit("gateway", "is another FRIDAY already running?")
        await stop.wait()


async def agent_loop(stop: asyncio.Event) -> None:
    """Placeholder agent loop: waits for shutdown. Real LLM/tool loop lives here."""
    await stop.wait()


# Well under the unit's TimeoutStopSec=15, so a stuck task is abandoned by us
# rather than resolved by SIGKILL -- which would skip echo-cancel cleanup.
SHUTDOWN_DRAIN_S = 6.0


async def supervise(stop: asyncio.Event) -> None:
    """Run voice_loop and agent_loop concurrently and independently.

    Each loop runs as its own task so a blocking failure in one does not
    stall the other. Any loop that raises is logged and the remaining
    loops keep running until shutdown is requested.
    """
    loops = {
        "voice_loop": asyncio.create_task(voice_loop(stop), name="voice_loop"),
        "agent_loop": asyncio.create_task(agent_loop(stop), name="agent_loop"),
        "gateway_loop": asyncio.create_task(gateway_loop(stop), name="gateway_loop"),
    }

    async def _watch(name: str, task: asyncio.Task) -> None:
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate failures per loop
            events.emit("crash", name, error=repr(exc))

    watchers = [asyncio.create_task(_watch(name, task)) for name, task in loops.items()]
    await stop.wait()

    for task in loops.values():
        task.cancel()

    # Bounded drain. `await task` after cancel() is not bounded by anything:
    # cancellation only takes effect at the next suspension point, and these
    # tasks run cleanup in `finally` blocks that touch the outside world --
    # closing a PortAudio capture stream, releasing module-echo-cancel through
    # pactl. Any one of those blocking leaves shutdown waiting forever, which
    # systemd resolves with SIGKILL after TimeoutStopSec. Reproduced: "State
    # 'stop-sigterm' timed out. Killing." with a pactl child killed alongside.
    #
    # SIGKILL is the bad outcome, not the hang itself: it skips the remaining
    # cleanup and leaves module-echo-cancel loaded, and a leaked module
    # renumbers every other application's capture device list. So the budget
    # here is deliberately well under TimeoutStopSec -- finishing cleanup
    # imperfectly on our own terms beats being killed mid-way through it.
    pending = [*loops.values(), *watchers]
    done, still_running = await asyncio.wait(pending, timeout=SHUTDOWN_DRAIN_S)
    for task in done:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
    if still_running:
        events.emit("shutdown", "abandoning tasks that would not stop",
                    count=len(still_running),
                    names=",".join(sorted(t.get_name() for t in still_running)),
                    after=f"{SHUTDOWN_DRAIN_S}s")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    def _request_stop() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Signal handlers via the event loop aren't available on all
            # platforms (e.g. Windows); fall back to default handling.
            pass


async def run() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop)
    await supervise(stop)


async def run_selftest(turns: int) -> None:
    """Drive `turns` synthetic turns through every span stage and write them."""
    for i in range(turns):
        turn_kind = "reflex" if turns == 1 or i % 3 != 0 else "reasoning"
        stages = REFLEX_STAGES if turn_kind == "reflex" else STAGES
        with start_turn(turn_kind=turn_kind) as t:
            for stage in stages:
                # tiny synthetic delay so offsets are non-decreasing but distinct
                await asyncio.sleep(random.uniform(0.0005, 0.003))
                t.mark(stage)
        if turns == 1:
            import json
            print(json.dumps(t.to_record(), sort_keys=True))


def _load_credentials() -> None:
    """Load ~/.friday/env into the environment if it is not already there.

    Done here, in the daemon's own startup, so every way of launching it
    behaves identically: `friday start`, `friday start --foreground`,
    `systemctl --user start friday`, and a bare `python -m friday`.

    This is not belt-and-braces with the unit's `EnvironmentFile=`. That
    directive does *not* understand `export KEY=value`, which is the format
    this file uses -- systemd parses the variable name as "export KEY" and
    silently drops it. Measured: a file containing `export FOO=bar` yields an
    empty $FOO. So under systemd the credentials never arrived at all, and the
    only symptom was `voice_loop=false` with no error anywhere.
    """
    from friday.daemon import load_env_file

    for key, value in load_env_file().items():
        os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    _load_credentials()
    parser = argparse.ArgumentParser(prog="friday")
    parser.add_argument("--selftest", action="store_true",
                         help="drive synthetic turn(s) through every span stage and exit")
    parser.add_argument("--turns", type=int, default=1,
                         help="number of synthetic turns to generate with --selftest")
    args = parser.parse_args(argv)

    if args.selftest:
        asyncio.run(run_selftest(args.turns))
        return 0

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())
    return 0
