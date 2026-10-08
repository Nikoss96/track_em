"""ProsodyAnalyzer на синтетических сигналах с известными параметрами и на TTS-речи."""

import pytest

from tests.harness import Harness

pytestmark = pytest.mark.slow


async def test_tone_f0_measured(fixtures):
    h = Harness(["prosody"])
    await h.run_file(fixtures / "tone.wav", with_vad=False)
    inst = [r for r in h.by("prosody") if r.payload.get("scope") == "instantaneous"]
    assert len(inst) >= 3
    for r in inst:
        assert abs(r.metric("f0_mean_hz").value - 200) < 2  # синус 200 Гц
        assert r.metric("voiced_ratio").value > 0.8


async def test_silence_is_insufficient(fixtures):
    h = Harness(["prosody"])
    await h.run_file(fixtures / "silence.wav", with_vad=False)
    rs = h.by("prosody")
    assert rs and all(r.status.value == "insufficient_data" for r in rs)
    assert all(r.metric("f0_mean_hz") is None and r.metric("f0_median_hz") is None for r in rs)  # F0 не выдумывается


async def test_tts_speech_window_features(fixtures):
    h = Harness(["prosody"])
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=20, with_vad=False)
    win = [r for r in h.by("prosody") if r.payload.get("scope") == "window" and r.status.value == "ok"]
    assert len(win) >= 5
    r = win[2]
    assert 120 < r.metric("f0_median_hz").value < 350
    assert r.metric("f0_sd_st").value > 0.5  # интонация не монотонна
    assert r.metric("intensity_sd_db").value > 1
    assert "egemaps" in r.payload and len(r.payload["egemaps"]) == 88
    assert r.metric("egemaps_hnr_db") is not None
    pauses = [p for x in h.by("prosody") for p in x.payload.get("pauses", [])]
    assert len(pauses) >= 3 and all(b - a >= 250 for a, b in pauses)
