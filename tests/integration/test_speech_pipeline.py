"""SpeechAnalyzer без Whisper: реальный Silero VAD + FakeASR на TTS-записи.

Проверяет сегментацию, принудительное разрезание с перекрытием, дедупликацию слов,
выравнивание с эталоном, WER только по прочитанной части и публикацию text.reference.
Качество распознавания реальной моделью проверяется в test_speech.py (slow, целевая машина).
"""

import soundfile as sf

from app.analyzers.speech import backends
from app.schemas.events import EventType
from tests.fakes import FakeASR
from tests.harness import Harness


async def _run(fixtures, monkeypatch, overrides, corrupt_every=0, max_s=None):
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    info = sf.info(str(fixtures / "reading_tts.wav"))
    FakeASR.reference, FakeASR.total_ms, FakeASR.corrupt_every = ref, info.duration * 1000, corrupt_every
    monkeypatch.setitem(backends.BACKENDS, "fake", FakeASR)
    ov = {"modules": {"speech": {"backend": "fake", "partial_interval_ms": 0, **overrides}}}
    h = Harness(["speech"], mode="reading", reference_text=ref, overrides=ov)
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=max_s)
    return h


async def test_segments_alignment_and_no_duplicates(fixtures, monkeypatch):
    h = await _run(fixtures, monkeypatch, {})
    finals = sorted([r for r in h.by("speech") if r.status.value == "ok" and not r.payload.get("summary")], key=lambda r: r.t_start_ms)
    assert len(finals) >= 3
    words = [w for r in finals for w in r.payload["words"]]
    texts = [w["w"] for w in words]
    assert len(words) == len({(w["s"], w["w"]) for w in words})  # без дублей
    assert all(b["s"] >= a["e"] for a, b in zip(words, words[1:], strict=False))
    cum = [r for r in h.by("speech") if r.payload.get("summary")][0].payload["cumulative"]
    # pipeline не вносит замен/вставок; пропуски — слова FakeASR, попавшие в паузы VAD
    assert cum["substitutions"] == 0 and cum["insertions"] == 0, cum
    assert cum["hits"] == len(texts) and cum["deletions"] < 0.25 * cum["ref_words"]
    refs = h.of_type(EventType.text_reference)
    assert refs and refs[0].sentence.startswith("На вокзале")
    tr = h.of_type(EventType.transcript_final)
    assert len(tr) == len({t.segment_id for t in tr})


async def test_forced_split_overlap_dedup(fixtures, monkeypatch):
    # короткий max_segment → принудительные разрезы с перекрытием 1 с
    h = await _run(fixtures, monkeypatch, {"max_segment_ms": 2500}, max_s=25)
    finals = sorted([r for r in h.by("speech") if r.status.value == "ok" and not r.payload.get("summary")], key=lambda r: r.t_start_ms)
    words = [w for r in finals for w in r.payload["words"]]
    assert len(words) == len({w["s"] for w in words})
    assert any("удалено дублей" in n for r in finals for n in r.quality.notes)


async def test_wer_counts_errors(fixtures, monkeypatch):
    h = await _run(fixtures, monkeypatch, {}, corrupt_every=5)
    cum = [r for r in h.by("speech") if r.payload.get("summary")][0].payload["cumulative"]
    n = cum["hits"] + cum["substitutions"]
    assert 0.12 < cum["substitutions"] / n < 0.3, cum  # ≈ каждое 5-е слово заменено
    acc = [r.metric("reading_accuracy") for r in h.by("speech") if r.metric("reading_accuracy")]
    assert acc and all(0 <= m.normalized_score <= 100 for m in acc)
