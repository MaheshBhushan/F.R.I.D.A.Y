"""FRIDAY audio device selection.

Playback always follows the system default sink. PipeWire therefore sends
FRIDAY to the internal speaker normally and to a headset when the desktop
selects one. Input remains pinned to the laptop microphone when available.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# Stable hardware endpoints used by input capture and the optional legacy
# echo-canceller. Normal playback deliberately does not use SPEAKER_SINK.
SPEAKER_SINK = (
    "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink"
)
MIC_SOURCE = (
    "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source"
)

# PortAudio device to open. With no PULSE_SINK, Pulse follows its default.
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
    """Use the system default output and the laptop mic when available."""
    sources = _names("sources")
    return Routing(
        None,
        MIC_SOURCE if MIC_SOURCE in sources else None,
        False,
        "system default output",
    )


def apply(routing: Optional[Routing] = None) -> Routing:
    """Apply input selection while leaving playback on the system default."""
    routing = routing or resolve()
    os.environ.pop("PULSE_SINK", None)
    if routing.source:
        os.environ["PULSE_SOURCE"] = routing.source
    else:
        os.environ.pop("PULSE_SOURCE", None)
    return routing


def output_device() -> str:
    return DEVICE


def input_device() -> str:
    return DEVICE
