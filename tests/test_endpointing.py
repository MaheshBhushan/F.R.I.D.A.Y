"""Deterministic adaptive endpointing state-machine tests."""

from __future__ import annotations

import pytest

from friday.voice.stt import (
    EndpointConfig,
    EndpointController,
    EndpointState,
    HardStopMatcher,
    SpeechSignal,
    TranscriptEvent,
    is_hard_stop_command,
)


def _config() -> EndpointConfig:
    return EndpointConfig(vad_pause_ms=400, fast_ms=650, patient_ms=1800, max_ms=2500)


@pytest.mark.parametrize("pause_ms", (700, 1200))
def test_thinking_pause_resumes_the_same_turn(pause_ms):
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(TranscriptEvent("I want you to check", False, turn_event="Update"), 0.1)
    endpoint.on_speech(SpeechSignal.STARTED, 0.1)
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    assert endpoint.state is EndpointState.POSSIBLE_END

    endpoint.on_speech(SpeechSignal.RESUMED, 0.6 + pause_ms / 1000)

    assert endpoint.state is EndpointState.SPEAKING
    assert endpoint.finalized_reason is None
    assert endpoint.stats.resumed_pauses == 1
    assert endpoint.stats.longest_pause_ms == pause_ms


def test_multiple_thinking_pauses_remain_one_turn():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(TranscriptEvent("Look at the project and", False, turn_event="Update"), 0.1)
    endpoint.on_speech(SpeechSignal.STARTED, 0.1)

    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    endpoint.on_speech(SpeechSignal.RESUMED, 1.2)
    endpoint.on_flux(
        TranscriptEvent(
            "Look at the project and see if the current implementation",
            False,
            turn_event="Update",
        ),
        1.3,
    )
    endpoint.on_speech(SpeechSignal.PAUSE, 2.0)
    endpoint.on_speech(SpeechSignal.RESUMED, 2.5)

    assert endpoint.state is EndpointState.SPEAKING
    assert endpoint.finalized_reason is None
    assert endpoint.stats.resumed_pauses == 2
    assert endpoint.stats.longest_pause_ms == 900


def test_true_end_uses_patient_fallback_when_flux_does_not_finish():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(
        TranscriptEvent(
            "Friday please check whether the backend is still running",
            False,
            turn_event="Update",
        ),
        0.1,
    )
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    assert endpoint.on_timeout(2.39) is None
    assert endpoint.on_timeout(2.4) == "patient_timeout"


def test_short_command_uses_fast_deadline():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(TranscriptEvent("Friday stop", False, turn_event="Update"), 0.1)
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    assert endpoint.deadline == pytest.approx(1.25)
    endpoint.on_timeout(1.25)
    assert endpoint.finalized_reason == "fast_timeout"


def test_short_but_incomplete_phrase_is_patient():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(TranscriptEvent("I want you to check", False, turn_event="Update"), 0.1)
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    assert endpoint.deadline == pytest.approx(2.4)
    assert endpoint.on_timeout(1.25) is None


def test_incomplete_phrase_uses_patient_deadline():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_flux(
        TranscriptEvent(
            "Friday I want you to look through the project and",
            False,
            turn_event="Update",
        ),
        0.1,
    )
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    assert endpoint.deadline == pytest.approx(2.4)


def test_flux_end_of_turn_is_authoritative():
    endpoint = EndpointController(_config(), clock=lambda: 0.0)
    endpoint.on_speech(SpeechSignal.PAUSE, 1.0)
    endpoint.on_flux(TranscriptEvent("done", True, speech_final=True, turn_event="EndOfTurn"), 1.1)
    assert endpoint.state is EndpointState.FINALIZED
    assert endpoint.finalized_reason == "flux_eot"


@pytest.mark.parametrize(
    "values",
    (
        {"vad_pause_ms": 0, "fast_ms": 650, "patient_ms": 1800, "max_ms": 2500},
        {"vad_pause_ms": 400, "fast_ms": 1800, "patient_ms": 650, "max_ms": 2500},
        {"vad_pause_ms": 400, "fast_ms": 650, "patient_ms": 2600, "max_ms": 2500},
    ),
)
def test_invalid_endpoint_config_fails_clearly(values):
    with pytest.raises(ValueError, match="endpoint"):
        EndpointConfig(**values)


@pytest.mark.parametrize("text", ("Friday stop", "Friday, stop.", "Hey Friday, stop"))
def test_isolated_hard_stop_matches(text):
    assert is_hard_stop_command(text)


@pytest.mark.parametrize(
    "text",
    (
        "Friday stop talking about Docker and explain Kubernetes",
        "What happens if I say Friday stop?",
        "The command is called Friday stop",
        "Don't use Friday stop",
        "The command Friday stop isn't working",
    ),
)
def test_hard_stop_does_not_substring_match(text):
    assert not is_hard_stop_command(text)


def test_hard_stop_requires_stable_streaming_evidence():
    matcher = HardStopMatcher()
    assert not matcher.feed(TranscriptEvent("Friday stop", False, turn_event="Update"))
    assert matcher.feed(TranscriptEvent("Friday stop", False, turn_event="Update"))
    matcher = HardStopMatcher()
    assert matcher.feed(TranscriptEvent("Friday stop", False, turn_event="EagerEndOfTurn"))
