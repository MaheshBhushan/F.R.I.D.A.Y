"""Audio Resource Manager: owns physical audio resources for the whole process.

Deliberately outside `friday.voice`. The voice modules are consumers of audio;
this package decides who is allowed to hold a device and in what state. It is
the intended home for microphone ownership, capture, and priority arbitration,
call detection and FRIDAY's listening state, so
that none of those live buried inside the wake-word or STT code.
"""

from friday.audio.capture import (  # noqa: F401
    AudioCaptureService,
    AudioFrame,
    AudioSubscription,
)

from friday.audio.manager import (  # noqa: F401
    AudioResourceManager,
    MicState,
)
from friday.audio.priority import (  # noqa: F401
    FRIDAY_PRIORITY,
    Owner,
    Priority,
    load_priorities,
    parse_owners,
    priority_of,
)
