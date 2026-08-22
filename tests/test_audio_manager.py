"""Audio Resource Manager tests.

The pactl fixtures are trimmed from real output on this machine, including the
echo-cancel virtual node: counting that node preempts FRIDAY forever, by her
own echo canceller, and nothing else would notice.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

from friday.audio import manager as mgr_mod
from friday.audio import priority as prio
from friday.audio.manager import AudioResourceManager, MicState

ECHO_CANCEL = '''Source Output #45
\tClient: n/a
\tProperties:
\t\tnode.name = "echo-cancel-capture"
\t\tnode.virtual = "true"
'''

FRIDAY = '''Source Output #2520
\tClient: 2512
\tProperties:
\t\tapplication.name = "ALSA plug-in [python3.11]"
\t\tapplication.process.id = "{pid}"
\t\tapplication.process.binary = "python3.11"
'''

ZOOM = '''Source Output #2600
\tClient: 2599
\tProperties:
\t\tapplication.name = "ZOOM VoiceEngine"
\t\tapplication.process.id = "9001"
\t\tapplication.process.binary = "zoom"
'''

UNKNOWN = '''Source Output #2700
\tClient: 2699
\tProperties:
\t\tapplication.name = "some-recorder"
\t\tapplication.process.id = "9100"
\t\tapplication.process.binary = "some-recorder"
'''


# -- priority ------------------------------------------------------------

def test_friday_is_the_lowest_tier():
    assert prio.FRIDAY_PRIORITY == max(prio.Priority)
    assert prio.Priority.P4_FRIDAY > prio.Priority.P3_INTERACTIVE


def test_echo_cancel_virtual_node_is_not_an_owner():
    assert prio.parse_owners(ECHO_CANCEL, own_pid=1234) == []


def test_own_stream_excluded_by_pid_not_by_name():
    text = ECHO_CANCEL + FRIDAY.format(pid=4242)
    assert prio.parse_owners(text, own_pid=4242) == []
    # PortAudio calls itself "ALSA plug-in [python3.11]" -- not unique, so a
    # different process with the same name must still count.
    assert len(prio.parse_owners(text, own_pid=1)) == 1


def test_call_app_lands_in_p1_and_preempts():
    owner = prio.parse_owners(ZOOM, own_pid=1)[0]
    assert owner.priority == prio.Priority.P1_CALLS
    assert owner.preempts_friday
    assert owner.tier == "P1_CALLS"


def test_unknown_app_is_p3_and_still_preempts():
    owner = prio.parse_owners(UNKNOWN, own_pid=1)[0]
    assert owner.priority == prio.Priority.P3_INTERACTIVE
    assert owner.preempts_friday


def test_dictation_apps_outrank_the_p3_default():
    # VoiceWin was silently landing in the P3 default. That still preempted
    # FRIDAY, so nothing looked broken -- but it ranked a live dictation take
    # below an idle browser tab. Pin the tier so it cannot drift back.
    for name in ("voicewin", "nerd-dictation"):
        assert prio.priority_of(name, name) == prio.Priority.P2_RECORDING


def test_strongest_match_wins_not_the_first():
    table = {"recorder": prio.Priority.P2_RECORDING,
             "some": prio.Priority.P1_CALLS}
    owner = prio.parse_owners(UNKNOWN, own_pid=1, table=table)[0]
    assert owner.priority == prio.Priority.P1_CALLS


def test_override_at_friday_tier_does_not_preempt():
    table = dict(prio.PRIORITY, **{"some-recorder": int(prio.Priority.P4_FRIDAY)})
    owner = prio.parse_owners(UNKNOWN, own_pid=1, table=table)[0]
    assert not owner.preempts_friday


# -- state machine -------------------------------------------------------

def _capture(chunks: int = 10_000):
    """Capture double: an endless frame source that records open/close."""
    log: list[str] = []

    @contextlib.asynccontextmanager
    async def opener() -> AsyncIterator[AsyncIterator[bytes]]:
        log.append("open")
        try:
            async def frames() -> AsyncIterator[bytes]:
                for _ in range(chunks):
                    await asyncio.sleep(0)
                    yield b"\x00" * 2560
            yield frames()
        finally:
            log.append("close")

    return opener, log


def _manager(text_ref, opener, **kw):
    async def _inspect():
        return text_ref()
    return AudioResourceManager(own_pid=4242, inspect=_inspect,
                                open_capture=opener, recheck_seconds=0.01, **kw)


def test_full_preempt_and_resume_cycle():
    """AVAILABLE -> FRIDAY_LISTENING -> PREEMPTING -> SUSPENDED -> LISTENING.

    The consumer drains in a background task on purpose: a suspension stops
    frames arriving, so anything awaiting the next frame inline would block
    forever -- which is precisely why the manager owns the stream and the
    consumer is left ignorant of ownership.
    """
    state = {"text": ECHO_CANCEL}
    opener, log = _capture()
    seen: list[MicState] = []
    preempts: list[int] = []
    forgets: list[int] = []
    m = _manager(lambda: state["text"], opener,
                 on_state=lambda s, o: seen.append(s),
                 on_preempt=lambda o: preempts.append(len(o)),
                 on_forget=lambda: forgets.append(1))

    async def _run():
        await m.refresh()
        stop = asyncio.Event()
        got = {"n": 0}

        async def _drain():
            async for _ in m.capture(stop):
                got["n"] += 1

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0.05)
        assert got["n"] > 0, "should be listening before anything else wants the mic"
        assert m.state is MicState.FRIDAY_LISTENING

        state["text"] = ECHO_CANCEL + ZOOM          # a call arrives
        await m.refresh()
        await asyncio.sleep(0.05)
        assert m.state is MicState.SUSPENDED
        assert preempts == [1], "preempt callback fires exactly once"
        assert forgets, "retained audio must be dropped on suspension"
        assert log.count("close") == 1, "the capture stream must be released"
        frozen = got["n"]
        await asyncio.sleep(0.05)
        assert got["n"] == frozen, "no frames may arrive while suspended"

        state["text"] = ECHO_CANCEL                 # the call ends
        await m.refresh()
        await asyncio.sleep(0.05)
        assert got["n"] > frozen, "frames must resume"
        assert m.state is MicState.FRIDAY_LISTENING
        assert log.count("open") == 2, "resume reopens the device"

        stop.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(_run())
    assert MicState.PREEMPTING in seen


def test_detector_is_armed_on_the_very_first_stream_not_just_on_resume():
    """The first capture stream must arm the wake warmup too.

    Regression: on_open did not exist, and begin_stream() was reached only via
    the preemption path. So the first stream of the process -- the one that
    drops its second frame, which openWakeWord scores as a wake word about a
    second later -- ran unguarded, and FRIDAY answered a phantom on every
    single startup. Only visible once the gateway could report turn counts
    during silence.
    """
    opens: list[str] = []
    opener, _log = _capture()
    manager = _manager(lambda: "", opener,
                       on_open=lambda: opens.append("armed"))

    async def _run():
        stop = asyncio.Event()
        seen = 0
        async for _ in manager.capture(stop):
            seen += 1
            if seen >= 2:
                stop.set()
                break
        assert opens == ["armed"], "warmup was never armed for the first stream"

    asyncio.run(_run())


def test_suspended_yields_no_frames():
    """SUSPENDED is deaf: not 'listening but not uploading'."""
    state = {"text": ECHO_CANCEL + ZOOM}          # taken from the start
    opener, log = _capture()
    m = _manager(lambda: state["text"], opener)

    async def _run():
        await m.refresh()
        stop = asyncio.Event()
        received = 0

        async def _drain():
            nonlocal received
            async for _ in m.capture(stop):
                received += 1

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0.05)
        assert m.state is MicState.SUSPENDED
        assert received == 0
        assert "open" not in log, "the mic must not even be opened"
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(_run())


def test_resume_reopens_the_stream():
    state = {"text": ECHO_CANCEL + ZOOM}
    opener, log = _capture()
    m = _manager(lambda: state["text"], opener)

    async def _run():
        await m.refresh()
        stop = asyncio.Event()
        seen = 0

        async def _drain():
            nonlocal seen
            async for _ in m.capture(stop):
                seen += 1

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0.03)
        assert seen == 0
        state["text"] = ECHO_CANCEL          # call ends
        await m.refresh()
        await asyncio.sleep(0.05)
        assert seen > 0, "frames must resume once the mic is free"
        assert log.count("open") == 1
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(_run())


def test_inspection_failure_fails_open():
    """A broken pactl must not strand FRIDAY deaf -- the worse outcome."""
    async def _inspect():
        raise OSError("pactl: not found")

    m = AudioResourceManager(own_pid=1, inspect=_inspect)

    async def _run():
        m.free.clear()
        m.taken.set()
        await m.refresh()
        assert m.free.is_set() and not m.taken.is_set()

    asyncio.run(_run())


def test_describe_names_the_highest_priority_owner():
    m = AudioResourceManager(own_pid=1)
    m.owners = prio.parse_owners(ZOOM + UNKNOWN, own_pid=1)
    text = m.describe()
    assert "zoom" in text and "P1_CALLS" in text
    assert "+1 more" in text
