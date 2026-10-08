"""VoiceEmotionAnalyzer без emotion2vec: реальный VAD + FakeSER."""

from app.analyzers.voice_emotion import backends
from app.analyzers.voice_emotion.analyzer import windows
from tests.fakes import FakeSER
from tests.harness import Harness


def test_windows():
    assert windows(0, 3000, 4000, 2000) == [(0, 3000)]
    assert windows(0, 9000, 4000, 2000) == [(0, 4000), (2000, 6000), (4000, 8000), (5000, 9000)]


async def test_segments_to_results(fixtures, monkeypatch):
    monkeypatch.setitem(backends.BACKENDS, "fake", FakeSER)
    h = Harness(["voice_emotion"], overrides={"modules": {"voice_emotion": {"backend": "fake", "min_speech_ms": 1500}}})
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=25)
    rs = h.by("voice_emotion")
    ok = [r for r in rs if r.status.value == "ok"]
    assert ok
    r = ok[0]
    assert r.metric("top_label").value == "neutral"
    assert r.metric("score_unk") is None  # служебный класс отброшен
    assert r.metric("top_label").confidence is None  # scores не выдаются за уверенность
    assert r.payload["validation"] == "not_validated_ru"
    assert r.payload["raw"][0]["embedding"][:3] == [0.0, 1.0, 2.0]
    assert all(r.t_end_ms - r.t_start_ms <= 4000 for r in ok)
    # короткие сегменты не классифицируются
    short = [x for x in rs if x.status.value == "insufficient_data"]
    assert all(x.t_end_ms - x.t_start_ms < 1500 for x in short)


async def test_unavailable_backend_isolated(fixtures):
    h = Harness(["voice_emotion", "prosody"], overrides={"modules": {"voice_emotion": {"backend": "nope"}}})
    await h.run_file(fixtures / "tone.wav", with_vad=False)
    states = {x.plugin.module_id: x.plugin.state.value for x in h.hosts}
    assert states["voice_emotion"] == "unavailable"
    assert h.by("prosody")  # остальные модули работают
