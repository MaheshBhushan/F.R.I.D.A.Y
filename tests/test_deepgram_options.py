"""Env-tunable Deepgram STT/TTS options reach the transports."""

from __future__ import annotations

from friday.voice import stt, tts


def test_stt_defaults(monkeypatch):
    monkeypatch.delenv("FRIDAY_STT_MODEL", raising=False)
    monkeypatch.delenv("FRIDAY_STT_KEYTERMS", raising=False)
    assert stt.env_stt_options() == {"model": stt.DEFAULT_STT_MODEL}


def test_stt_model_and_keyterms(monkeypatch):
    monkeypatch.setenv("FRIDAY_STT_MODEL", "flux-general-en")
    monkeypatch.setenv("FRIDAY_STT_KEYTERMS", "tmux, pacman,Hermes,, ")
    assert stt.env_stt_options() == {
        "model": "flux-general-en",
        "keyterm": ["tmux", "pacman", "Hermes"],
    }


def test_pool_forwards_options_to_transport(monkeypatch):
    monkeypatch.setenv("FRIDAY_STT_MODEL", "flux-general-en")
    monkeypatch.setenv("FRIDAY_STT_KEYTERMS", "tmux")
    seen = {}

    class Fake:
        def __init__(self, api_key, **config):
            seen.update(config)

    pool = stt.FluxTransportPool("k", transport_factory=Fake, **stt.env_stt_options())
    pool._transport_factory("k", **pool._config)
    assert seen["model"] == "flux-general-en" and seen["keyterm"] == ["tmux"]


def test_tts_voice(monkeypatch):
    monkeypatch.delenv("FRIDAY_TTS_VOICE", raising=False)
    assert tts.env_tts_options() == {"model": tts.DEFAULT_TTS_VOICE}
    monkeypatch.setenv("FRIDAY_TTS_VOICE", "aura-2-thalia-en")
    assert tts.env_tts_options() == {"model": "aura-2-thalia-en"}
