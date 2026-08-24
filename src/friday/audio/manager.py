"""The microphone as an explicit state machine.

    AVAILABLE
       |
       v
    FRIDAY_LISTENING
       |  higher-priority request
       v
    PREEMPTING  -- invalidate incomplete turn, close STT, release audio
       |
       v
    SUSPENDED
       |  higher-priority owner releases
       v
    AVAILABLE -> FRIDAY_LISTENING

SUSPENDED means genuinely deaf, not "listening locally but not uploading":
the capture stream is closed, so no wake-word inference runs, no STT stream
exists, and `on_forget` is invoked so buffered audio (the wake-word pre-roll
ring, in particular) is dropped rather than kept across the suspension. A
1.5-second rolling buffer of the room retained while someone else is on a call
is exactly the thing a voice assistant must not do.

Preemption is immediate. A higher-priority application does not wait for FRIDAY
to finish a turn, and FRIDAY does not fight for the device or retry acquisition:
she waits to be told the microphone is free again. The periodic re-check exists
only so a dead `pactl subscribe` cannot strand her in SUSPENDED forever; it is
a read-only inspection, never an acquisition attempt.

The manager owns the capture stream so that FRIDAY's microphone subsystem is
disposable: it can be torn down and rebuilt while the brain, memory, world
state and coding agents keep running untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
from enum import Enum
from typing import AsyncIterator, Awaitable, Callable, Optional

from friday.audio.echocancel import EchoCancelModule
from friday.audio.priority import FRIDAY_PRIORITY, Owner, parse_owners

# Read-only re-inspection interval. Insurance against a dead subscription,
# not a reacquisition attempt.
RECHECK_SECONDS = 5.0


class MicState(Enum):
    AVAILABLE = "available"
    FRIDAY_LISTENING = "friday_listening"
    PREEMPTING = "preempting"
    SUSPENDED = "suspended"


async def _pactl(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "pactl", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode("utf-8", "replace")


class AudioResourceManager:
    """Arbitrates microphone ownership and drives the state machine.

    Callbacks are optional. State/open/forget callbacks are synchronous;
    preemption may return an awaitable so transport cleanup completes before
    the physical capture stream is released:

    * `on_state` -- every state change, for the UI/indicator.
    * `on_preempt` -- fired in PREEMPTING, before the stream closes. This is
      where an incomplete turn is invalidated.
    * `on_forget` -- fired once the stream is closed, to drop retained audio.
    """

    def __init__(
        self,
        *,
        own_pid: Optional[int] = None,
        open_capture: Optional[Callable[[], "contextlib.AbstractAsyncContextManager"]] = None,
        inspect: Optional[Callable[[], Awaitable[str]]] = None,
        recheck_seconds: float = RECHECK_SECONDS,
        on_state: Optional[Callable[["MicState", list[Owner]], None]] = None,
        on_preempt: Optional[Callable[[list[Owner]], Optional[Awaitable[None]]]] = None,
        on_forget: Optional[Callable[[], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
        echo_cancel: Optional[EchoCancelModule] = None,
        manage_echo_cancel: bool = False,
    ) -> None:
        self._own_pid = own_pid if own_pid is not None else os.getpid()
        self._open_capture = open_capture
        self._inspect = inspect
        self._recheck = recheck_seconds
        self._on_state = on_state
        self._on_preempt = on_preempt
        self._on_forget = on_forget
        self._on_open = on_open

        self.state = MicState.AVAILABLE
        self.echo_status = None
        self.owners: list[Owner] = []
        self.free = asyncio.Event()
        self.free.set()
        self.taken = asyncio.Event()
        self._watch_task: Optional[asyncio.Task] = None
        # Loaded on demand so the echo canceller's nodes append to the END of
        # the device list instead of shifting every other capture device's
        # index. See friday.audio.echocancel for the incident that motivated it.
        self._echo = echo_cancel if echo_cancel is not None else (
            EchoCancelModule() if manage_echo_cancel else None)

    # -- observation ------------------------------------------------------

    @property
    def blockers(self) -> list[Owner]:
        return [o for o in self.owners if o.preempts_friday]

    def set_callbacks(
        self,
        *,
        on_state: Optional[Callable[["MicState", list[Owner]], None]] = None,
        on_preempt: Optional[Callable[[list[Owner]], Optional[Awaitable[None]]]] = None,
        on_forget: Optional[Callable[[], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
    ) -> None:
        """Bind the lifecycle owner when a manager is injected."""
        self._on_state = on_state
        self._on_preempt = on_preempt
        self._on_forget = on_forget
        self._on_open = on_open

    def describe(self) -> str:
        """The line to put in front of the user."""
        blockers = self.blockers
        if not blockers:
            return "microphone free"
        worst = min(blockers, key=lambda o: o.priority)
        extra = f" (+{len(blockers) - 1} more)" if len(blockers) > 1 else ""
        return f"in use by {worst.label} [{worst.tier}]{extra}"

    async def refresh(self) -> list[Owner]:
        """Re-read ownership and update `free`/`taken`. Returns blockers.

        Fails OPEN: a broken `pactl` must not leave FRIDAY permanently deaf,
        which is both the worse outcome and the harder one to diagnose.
        """
        try:
            text = await (self._inspect() if self._inspect is not None
                          else _pactl("list", "source-outputs"))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.owners = []
            self.taken.clear()
            self.free.set()
            return []
        self.owners = parse_owners(text, own_pid=self._own_pid)
        blockers = self.blockers
        if blockers:
            self.free.clear()
            self.taken.set()
        else:
            self.taken.clear()
            self.free.set()
        return blockers

    async def _watch(self) -> None:
        await self.refresh()
        while True:
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "pactl", "subscribe",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                assert proc.stdout is not None
                while True:
                    read = asyncio.create_task(proc.stdout.readline())
                    tick = asyncio.create_task(asyncio.sleep(self._recheck))
                    try:
                        done, _ = await asyncio.wait(
                            {read, tick}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for t in (read, tick):
                            if not t.done():
                                t.cancel()
                        await asyncio.gather(read, tick, return_exceptions=True)
                    if read in done:
                        line = await read
                        if not line:
                            break
                        # Client events matter too: a stream's properties can
                        # land after the source-output event announcing it.
                        if b"source-output" not in line and b"client" not in line:
                            continue
                    await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.taken.clear()
                self.free.set()          # fail open
            finally:
                if proc is not None and proc.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        proc.terminate()
                    with contextlib.suppress(Exception):
                        await proc.wait()
            await asyncio.sleep(self._recheck)

    async def start(self) -> None:
        if self._echo is not None:
            status = await self._echo.ensure_loaded()
            self.echo_status = status
            if not status.available:
                print(f"[friday] no echo cancellation ({status.reason}); "
                      "running on the raw microphone", flush=True)
        if self._watch_task is None or self._watch_task.done():
            self._watch_task = asyncio.create_task(self._watch())
            await self.refresh()

    async def stop(self) -> None:
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None
        if self._echo is not None:
            # Leaves the device list exactly as it was found, so nothing else
            # on the machine gets renumbered by FRIDAY having run.
            await self._echo.release()

    # -- state machine ----------------------------------------------------

    def _enter(self, state: MicState) -> None:
        if state is self.state:
            return
        self.state = state
        if self._on_state is not None:
            with contextlib.suppress(Exception):
                self._on_state(state, list(self.owners))

    async def capture(self, stop: asyncio.Event) -> AsyncIterator[bytes]:
        """Yield microphone frames for as long as FRIDAY is entitled to them.

        Frames simply stop arriving across a suspension and resume afterwards,
        so a consumer needs no knowledge of ownership at all -- which is the
        point of making the microphone subsystem disposable.
        """
        try:
            while not stop.is_set():
                if not self.free.is_set():
                    self._enter(MicState.SUSPENDED)
                    await _race(self.free.wait(), stop.wait())
                    if stop.is_set():
                        break
                self._enter(MicState.AVAILABLE)
                async for chunk in self._one_session(stop):
                    yield chunk
        finally:
            self._enter(MicState.AVAILABLE)

    async def _one_session(self, stop: asyncio.Event) -> AsyncIterator[bytes]:
        """One held-microphone session: open, stream, release on preemption."""
        opener = self._open_capture or _sounddevice_capture
        preempted = False
        try:
            async with opener() as frames:
                # Every freshly opened capture stream needs the wake detector
                # re-armed, including the very first one. Firing this only on
                # resume left the first stream of the process unguarded --
                # exactly the one that drops its second frame, which openWakeWord
                # scores as a wake word about a second later. That produced a
                # phantom turn on every startup.
                if self._on_open is not None:
                    with contextlib.suppress(Exception):
                        self._on_open()
                self._enter(MicState.FRIDAY_LISTENING)
                while True:
                    nxt = asyncio.create_task(_anext(frames))
                    lost = asyncio.create_task(self.taken.wait())
                    halt = asyncio.create_task(stop.wait())
                    waiters = {nxt, lost, halt}
                    try:
                        done, _ = await asyncio.wait(
                            waiters, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for t in waiters:
                            if not t.done():
                                t.cancel()
                        await asyncio.gather(*waiters, return_exceptions=True)
                    if self.taken.is_set():
                        # Immediate: the higher-priority app does not wait for
                        # FRIDAY to finish, and the incomplete turn dies here
                        # rather than being resumed twenty minutes later.
                        preempted = True
                        self._enter(MicState.PREEMPTING)
                        if self._on_preempt is not None:
                            try:
                                result = self._on_preempt(self.blockers)
                                if inspect.isawaitable(result):
                                    await result
                            except Exception:
                                pass
                        return
                    if stop.is_set():
                        return
                    chunk = await nxt
                    if chunk is None:
                        return
                    yield chunk
        finally:
            # The stream is closed by now, so nothing can be recorded. Drop
            # whatever was buffered: no audio is retained across a suspension.
            if self._on_forget is not None:
                with contextlib.suppress(Exception):
                    self._on_forget()
            if preempted:
                self._enter(MicState.SUSPENDED)


async def _anext(it: AsyncIterator[bytes]) -> Optional[bytes]:
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return None


async def _race(*coros) -> None:
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@contextlib.asynccontextmanager
async def _sounddevice_capture(sample_rate: int = 16000,
                               blocksize: int = 1280) -> AsyncIterator[AsyncIterator[bytes]]:
    """Default capture: a PortAudio input stream as an async frame iterator."""
    import sounddevice as sd

    from friday.voice import devices

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[bytes]" = asyncio.Queue()

    def _callback(indata, count, time_info, status) -> None:
        # Audio thread: hand off and return. Nothing expensive here.
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    devices.apply()
    stream = sd.RawInputStream(
        samplerate=sample_rate, blocksize=blocksize, dtype="int16",
        channels=1, callback=_callback, device=devices.input_device(),
    )

    async def _frames() -> AsyncIterator[bytes]:
        while True:
            yield await queue.get()

    with stream:
        yield _frames()


async def _main() -> int:
    from friday.audio.priority import PRIORITY_PATH, Priority, load_priorities

    table = load_priorities()
    owners = parse_owners(await _pactl("list", "source-outputs"),
                          own_pid=os.getpid(), table=table)
    print("microphone priority tiers (lower wins):")
    for p in Priority:
        mark = "  <- friday" if p is FRIDAY_PRIORITY else ""
        print(f"  {p.value}  {p.name}{mark}")
    if PRIORITY_PATH.exists():
        print(f"overrides: {PRIORITY_PATH}")
    if not owners:
        print("\nno other application is capturing -- friday may listen")
        return 0
    print("\ncapturing now:")
    for o in owners:
        verdict = "PREEMPTS friday" if o.preempts_friday else "shares with friday"
        print(f"  #{o.index:<5} {o.label:<26} pid={o.pid:<8} "
              f"{o.tier:<15} {verdict}")
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
