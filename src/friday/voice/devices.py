"""Audio routing for FRIDAY's own voice.

FRIDAY always speaks through the built-in speaker and listens through the
built-in mic, regardless of where the rest of the system's audio is going. The
user's music, browser and everything else keep following the normal default
sink; only this process is pinned.

Two reasons it has to be this way:

  * Acoustic echo cancellation is only meaningful on the speaker path. There is
    no echo to cancel when output goes to headphones, and pinning the AEC module
    to a moving default sink means it silently follows audio to HDMI where no
    acoustic loop exists at all.
  * A voice assistant that goes silent because the user plugged in headphones or
    a monitor is not an assistant.

Mechanism: PortAudio exposes only ALSA plugin names (`pipewire`, `pulse`,
`default`), not individual sinks, so the stream cannot name a target device
directly. The Pulse backend does honour `PULSE_SINK` / `PULSE_SOURCE`, which are
independent for output and input and scoped to this process's environment -- so
they pin FRIDAY without touching any other application.

`apply()` must run BEFORE the first stream is opened; the backend reads the
environment at stream-open time.

Hardware caveat on this machine: the card's profiles are
    HiFi (..., Headphones, Headset, Mic1)
    HiFi (..., Headphones, Mic1, Mic2)
    HiFi (..., Headset, Mic1, Speaker)
    HiFi (..., Mic1, Mic2, Speaker)
and NO profile contains both Speaker and Headphones -- they are mutually
exclusive ports on a shared DAC path. So "FRIDAY on the speaker while music
plays on wired headphones" is not achievable here. Speaker alongside
HDMI/DisplayPort, Bluetooth or USB output is fine, since those coexist with
Speaker in every profile. With wired headphones in, `resolve()` reports
`degraded` and FRIDAY falls back to the default sink rather than going mute.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# The AEC nodes from ~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf,
# themselves pinned to the built-in speaker and mic.
EC_SINK = "echo-cancel-sink"
EC_SOURCE = "echo-cancel-source"

# Direct fallbacks if the echo-cancel module is not loaded.
SPEAKER_SINK = (
    "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink"
)
MIC_SOURCE = (
    "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source"
)

# PortAudio device to open. The Pulse backend is what reads PULSE_SINK/SOURCE.
DEVICE = "pulse"


@dataclass
class Routing:
    sink: Optional[str]
    source: Optional[str]
    degraded: bool
    reason: str


def _names(kind: str) -> set[str]:
    """Sink or source node names currently present, or an empty set."""
    try:
        out = subprocess.run(
            ["pactl", "list", "short", kind],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    return {
        line.split("\t")[1]
        for line in out.stdout.splitlines()
        if len(line.split("\t")) > 1
    }


def resolve() -> Routing:
    """Pick FRIDAY's sink and source, preferring the echo-cancelled pair."""
    sinks, sources = _names("sinks"), _names("sources")

    if EC_SINK in sinks and EC_SOURCE in sources:
        return Routing(EC_SINK, EC_SOURCE, False, "echo-cancelled speaker path")

    # AEC absent: use the raw speaker/mic. Playback is still on the speaker, so
    # mic gating during speech is the only self-interruption defence.
    if SPEAKER_SINK in sinks:
        source = MIC_SOURCE if MIC_SOURCE in sources else None
        return Routing(
            SPEAKER_SINK,
            source,
            True,
            "echo-cancel module not loaded; raw speaker, no AEC",
        )

    # No speaker at all -- the active card profile is a Headphones one. Fall
    # back to the system default rather than going mute.
    return Routing(
        None,
        MIC_SOURCE if MIC_SOURCE in sources else None,
        True,
        "no speaker sink in the active card profile (wired headphones?); "
        "using the default sink",
    )


def apply(routing: Optional[Routing] = None) -> Routing:
    """Pin this process's audio. Must be called before any stream is opened."""
    routing = routing or resolve()
    if routing.sink:
        os.environ["PULSE_SINK"] = routing.sink
    if routing.source:
        os.environ["PULSE_SOURCE"] = routing.source
    return routing


def output_device() -> str:
    return DEVICE


def input_device() -> str:
    return DEVICE
