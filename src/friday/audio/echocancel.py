"""Load the echo canceller only while FRIDAY is running.

It used to live in `~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf`,
loaded at PipeWire startup and present 24/7 for a process that runs minutes a
day. That had a real cost beyond tidiness, and it is the reason this module
exists:

**Modules loaded from `pipewire.conf.d` load BEFORE the ALSA devices exist**, so
they get lower node IDs and sort to the FRONT of the device list. Every
pre-existing capture device shifts down by two. Any application that remembers
its microphone by list index -- which is a common and awful pattern -- silently
starts recording something else. Observed for real: VoiceWin had
`InputDeviceNumber: 1`, that position became `echo-cancel-sink.monitor` (an
output monitor), and it reported "No audio captured" while the microphone was
perfectly healthy.

Loading at runtime inverts that. The nodes get higher IDs than the hardware and
append to the END of the list, so nothing else is ever renumbered. Measured:
with the module loaded at runtime, Mic1 stays at index 5 whether FRIDAY is
running or not.

Failure is soft by design. If the module will not load, FRIDAY runs on the raw
microphone -- `devices.resolve()` already reports that as degraded -- because no
echo cancellation is far better than no assistant.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from friday.voice import devices

SOURCE_NAME = "echo-cancel-source"
SINK_NAME = "echo-cancel-sink"

# Same AEC tuning the config file used. noise_suppression and gain_control stay
# off: both are non-linear and were measured to hurt wake-word scores more than
# the residual echo they remove.
AEC_ARGS = (
    "webrtc.extended_filter=true "
    "webrtc.high_pass_filter=true "
    "webrtc.noise_suppression=false "
    "webrtc.gain_control=false "
    "webrtc.delay_agnostic=true"
)

# The node takes a moment to appear after load-module returns.
APPEAR_TIMEOUT_S = 4.0
APPEAR_POLL_S = 0.2

_MODULE_ID_RE = re.compile(r"^\s*(\d+)\s*$")

PactlRunner = Callable[..., Awaitable[tuple[int, str]]]

from friday.core import events


@dataclass
class Status:
    """Outcome of `ensure_loaded`."""

    available: bool          # is an echo-cancelled source usable?
    owned: bool              # did WE load it (and must therefore unload it)?
    module_id: Optional[int]
    reason: str


# Every pactl call is bounded. Unbounded was a real, reproduced hang: on
# shutdown `pactl unload-module` can block indefinitely inside the PipeWire
# server round-trip while a capture stream is still attached to the module.
# The daemon then sat in communicate() until systemd's TimeoutStopSec expired
# and SIGKILLed both it and the pactl child (journal, 2026-08-22: "Killing
# process ... (pactl) with signal SIGKILL"), which left module-echo-cancel
# loaded -- and a leaked module renumbers every other application's device
# list, which is the bug that silently cost Chrome its microphone.
#
# Timing out here does not lose the unload: reap_echo_cancel() in daemon.py
# still sweeps a leftover module after a hard kill. This just stops one
# uncooperative IPC call from holding the whole shutdown hostage.
PACTL_TIMEOUT_S = 5.0


async def _run_pactl(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "pactl", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=PACTL_TIMEOUT_S)
    except asyncio.TimeoutError:
        # Kill, don't terminate: a pactl wedged on a server round-trip is not
        # reliably responsive to SIGTERM, and orphaning it would leave a child
        # for systemd to shoot down later -- the exact symptom being fixed.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        events.emit("audio", "pactl timed out", cmd=" ".join(args),
                    after=f"{PACTL_TIMEOUT_S}s")
        return 1, ""
    return proc.returncode or 0, out.decode("utf-8", "replace")


class EchoCancelModule:
    """Owns the lifetime of `module-echo-cancel` for this process."""

    def __init__(
        self,
        *,
        source: str = SOURCE_NAME,
        sink: str = SINK_NAME,
        mic: str = devices.MIC_SOURCE,
        speaker: str = devices.SPEAKER_SINK,
        pactl: Optional[PactlRunner] = None,
        appear_timeout: float = APPEAR_TIMEOUT_S,
    ) -> None:
        self._source = source
        self._sink = sink
        self._mic = mic
        self._speaker = speaker
        self._pactl = pactl or _run_pactl
        self._appear_timeout = appear_timeout
        self.status = Status(False, False, None, "not loaded")

    async def _sources(self) -> str:
        _, out = await self._pactl("list", "short", "sources")
        return out

    async def _present(self) -> bool:
        return self._source in await self._sources()

    async def ensure_loaded(self) -> Status:
        """Make an echo-cancelled source available, loading it if needed."""
        try:
            if await self._present():
                # Someone else owns it -- a leftover config file, or a second
                # FRIDAY. Use it, but never unload what we did not load.
                self.status = Status(True, False, None,
                                     "already present; not owned by this process")
                return self.status

            code, out = await self._pactl(
                "load-module", "module-echo-cancel",
                f"source_name={self._source}",
                f"sink_name={self._sink}",
                f"source_master={self._mic}",
                f"sink_master={self._speaker}",
                "aec_method=webrtc",
                f"aec_args={AEC_ARGS}",
            )
            if code != 0:
                self.status = Status(False, False, None,
                                    f"load-module failed: {out.strip()[:120]}")
                return self.status

            module_id = None
            match = _MODULE_ID_RE.match(out.strip().splitlines()[-1] if out.strip() else "")
            if match:
                module_id = int(match.group(1))

            # Pinning the masters is the point: an unpinned AEC follows the
            # DEFAULT sink, so it silently ends up cancelling against HDMI
            # where no acoustic loop exists at all.
            deadline = asyncio.get_running_loop().time() + self._appear_timeout
            while asyncio.get_running_loop().time() < deadline:
                if await self._present():
                    self.status = Status(True, True, module_id, "loaded on demand")
                    return self.status
                await asyncio.sleep(APPEAR_POLL_S)

            # Loaded but never appeared: unload so we do not leak a half-built
            # module across restarts.
            if module_id is not None:
                await self._pactl("unload-module", str(module_id))
            self.status = Status(False, False, None,
                                 "module loaded but the source never appeared")
            return self.status
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - degraded audio beats no assistant
            self.status = Status(False, False, None, f"{type(exc).__name__}: {exc}")
            return self.status

    async def release(self) -> None:
        """Unload the module, but only if this process loaded it."""
        if not (self.status.owned and self.status.module_id is not None):
            return
        module_id = self.status.module_id
        self.status = Status(False, False, None, "released")
        try:
            await self._pactl("unload-module", str(module_id))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
