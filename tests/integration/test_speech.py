"""SpeechAnalyzer: faster-whisper (модель активного профиля) на русской TTS-речи с известным эталоном (slow)."""

import pytest

from app.schemas.events import EventType
from tests.harness import Harness

pytestmark = pytest.mark.slow


async def test_reading_asr_alignment(fixtures):
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    h = Harness(["speech"], mode="reading", reference_text=ref, overrides={"modules": {"speech": {"partial_interval_ms": 0}}})
    await h.run_file(fixtures / "reading_tts.wav")
    finals = [r for r in h.by("speech") if r.finality == "final" and r.status.value == "ok" and not r.payload.get("summary")]
    assert len(finals) >= 3
    # отсутствие дублирования: сегменты и слова не перекрываются по времени
    words = [w for r in sorted(finals, key=lambda r: r.t_start_ms) for w in r.payload["words"]]
    assert all(b["s"] >= a["e"] - 60 for a, b in zip(words, words[1:], strict=False))
    tr = h.of_type(EventType.transcript_final)
    assert len({t.segment_id for t in tr}) == len(tr)
    text = " ".join(r.payload["text"] for r in finals).lower()
    assert "вокзал" in text and "тонкий" in text
    summary = [r for r in h.by("speech") if r.payload.get("summary")][0]
    cum = summary.payload["cumulative"]
    assert cum["ref_words"] >= 60
    assert cum["wer"] < 0.35, cum  # чистая синтетическая речь
    assert cum["progress"] > 0.8
    refs = h.of_type(EventType.text_reference)
    assert refs and all(r.sentence for r in refs)
    print("WER", cum)
