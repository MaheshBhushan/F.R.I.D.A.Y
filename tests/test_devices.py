from friday.voice import devices


def test_playback_always_uses_system_default(monkeypatch):
    monkeypatch.setenv("PULSE_SINK", "old-pinned-speaker")
    monkeypatch.setattr(devices, "_names", lambda kind: {devices.MIC_SOURCE})

    routing = devices.apply()

    assert routing.sink is None
    assert "PULSE_SINK" not in devices.os.environ
    assert devices.output_device() == "pulse"
