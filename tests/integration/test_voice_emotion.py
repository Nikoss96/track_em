"""VoiceEmotionAnalyzer на реальной модели emotion2vec+ (slow; запускать на целевой машине).

Проверяется работоспособность адаптера, а НЕ корректность эмоций: TTS-речь синтетическая,
модель на русском не валидирована.
"""

import pytest

from tests.harness import Harness

pytestmark = pytest.mark.slow


async def test_emotion2vec_real(fixtures):
    h = Harness(["voice_emotion"])
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=30)
    host = h.hosts[0]
    if host.plugin.state.value == "unavailable":
        pytest.skip(f"emotion2vec недоступен: {host.plugin.detail}")
    ok = [r for r in h.by("voice_emotion") if r.status.value == "ok"]
    assert ok
    r = ok[0]
    scores = r.payload["scores"]
    assert set(scores) >= {"angry", "happy", "neutral", "sad"}
    assert 0.9 < sum(scores.values()) <= 1.01
    assert r.payload["embedding_dim"] in (768, 1024)
    assert r.payload["validation"] == "not_validated_ru"
