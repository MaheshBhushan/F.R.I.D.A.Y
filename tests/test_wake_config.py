"""The wake model is configurable because the only two pretrained models are
"alexa" and "hey_jarvis" -- neither of which is "friday"."""

import pytest

from friday.voice import wake


def test_default_model_is_alexa(monkeypatch):
    monkeypatch.delenv("FRIDAY_WAKE_MODEL", raising=False)
    assert wake.default_models() == ("alexa",)


def test_env_overrides_the_model(monkeypatch):
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "hey_jarvis")
    assert wake.default_models() == ("hey_jarvis",)


def test_env_accepts_several_models(monkeypatch):
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "alexa, hey_jarvis")
    assert wake.default_models() == ("alexa", "hey_jarvis")


def test_blank_env_falls_back_to_the_default(monkeypatch):
    # An empty or whitespace-only value is a user who unset it badly, not a
    # request for zero wake models -- which would make her permanently deaf.
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "   ")
    assert wake.default_models() == ("alexa",)


def test_detector_reports_the_models_it_loaded(monkeypatch):
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "hey_jarvis")
    detector = wake.WakeWordDetector()
    assert detector.model_names == ("hey_jarvis",)


def test_explicit_argument_beats_the_env(monkeypatch):
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "hey_jarvis")
    detector = wake.WakeWordDetector(model_names=("alexa",))
    assert detector.model_names == ("alexa",)


# ------------------------------------------- custom trained models

def test_a_bundled_name_is_passed_through_untouched():
    # openwakeword resolves its own names against its bundled resources; if we
    # rewrote those into paths, "alexa" would stop working.
    assert wake.resolve_models(("alexa",)) == ("alexa",)


def test_an_unknown_name_is_passed_through_not_swallowed():
    # A typo must fail loudly inside Model(), not be silently dropped here --
    # a dropped name means zero wake models, i.e. permanently deaf.
    assert wake.resolve_models(("nonexistent_xyz",)) == ("nonexistent_xyz",)


def test_a_local_model_becomes_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "MODELS_DIR", tmp_path)
    (tmp_path / "friday.onnx").write_bytes(b"stub")
    assert wake.resolve_models(("friday",)) == (str(tmp_path / "friday.onnx"),)


def test_an_explicit_path_is_honoured(monkeypatch, tmp_path):
    # Lets a freshly trained model be A/B'd before it is installed.
    monkeypatch.setattr(wake, "MODELS_DIR", tmp_path)
    assert wake.resolve_models(("/tmp/x/hey_friday.onnx",)) == ("/tmp/x/hey_friday.onnx",)


def test_mixed_local_and_bundled_resolve_independently(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "MODELS_DIR", tmp_path)
    (tmp_path / "friday.onnx").write_bytes(b"stub")
    assert wake.resolve_models(("friday", "alexa")) == (
        str(tmp_path / "friday.onnx"), "alexa")


def test_available_models_lists_local_only(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "MODELS_DIR", tmp_path)
    (tmp_path / "friday.onnx").write_bytes(b"stub")
    (tmp_path / "hey_friday.onnx").write_bytes(b"stub")
    (tmp_path / "notes.txt").write_text("ignored")
    assert wake.available_models() == ("friday", "hey_friday")


def test_available_models_is_empty_when_dir_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(wake, "MODELS_DIR", tmp_path / "nope")
    assert wake.available_models() == ()


def test_detector_reports_friendly_names_not_paths(monkeypatch):
    # The log line and `friday hear` header show these; a 90-char absolute path
    # on an NTFS mount with spaces in it makes both unreadable.
    monkeypatch.setenv("FRIDAY_WAKE_MODEL", "hey_jarvis")
    assert wake.WakeWordDetector().model_names == ("hey_jarvis",)


def test_raw_buffer_is_right_sized():
    """openWakeWord keeps 10s of raw audio (160,000 Python ints) but only ever
    reads the newest ~1,760 samples, re-materialising the whole deque every
    80ms frame. Profiled, that single line was 25% of the daemon's idle CPU.
    We shrink it in __init__; this guards the shrink against an upstream
    rename silently restoring the cost."""
    from friday.voice.wake import RAW_BUFFER_SECONDS, SAMPLE_RATE, WakeWordDetector

    detector = WakeWordDetector()
    buffer = detector._model.preprocessor.raw_data_buffer
    assert buffer.maxlen == int(SAMPLE_RATE * RAW_BUFFER_SECONDS)

    # Must still comfortably exceed the widest slice openwakeword takes
    # (n_samples + 160*3 for an 80ms frame), or we would corrupt detection.
    assert buffer.maxlen > 1280 + 160 * 3
