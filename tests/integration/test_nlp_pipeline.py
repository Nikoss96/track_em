"""TextAnalyzer + TransactionalAnalyzer с поддельным Ollama (без сети и без LLM)."""

import soundfile as sf

from app.analyzers import llm_client
from app.analyzers.speech import backends as asr_backends
from app.analyzers.transactional.analyzer import has_direct_speech
from app.config import load_config
from tests.fakes import FakeASR, FakeOllama
from tests.harness import Harness


def _setup(fixtures, monkeypatch, mode="ok"):
    fake = FakeOllama(mode, model=load_config("auto")["llm"]["model"])
    monkeypatch.setattr(llm_client, "TRANSPORT", fake.transport)
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    FakeASR.reference, FakeASR.total_ms, FakeASR.corrupt_every = ref, sf.info(str(fixtures / "reading_tts.wav")).duration * 1000, 0
    monkeypatch.setitem(asr_backends.BACKENDS, "fake", FakeASR)
    return fake, ref


OV = {"modules": {"speech": {"backend": "fake", "partial_interval_ms": 0}}}


def test_direct_speech_detection():
    assert has_direct_speech("— Порфирий! — воскликнул толстый.")
    assert has_direct_speech("Он сказал: «Иди сюда».")
    assert not has_direct_speech("На вокзале встретились два приятеля: один толстый, другой тонкий.")


async def test_reading_mode_reference_fragments(fixtures, monkeypatch):
    fake, ref = _setup(fixtures, monkeypatch)
    h = Harness(["speech", "text", "transactional"], mode="reading", reference_text=ref, overrides=OV)
    await h.run_file(fixtures / "reading_tts.wav")
    text = h.by("text")
    assert text and all(r.status.value == "ok" for r in text)
    r = text[0]
    assert r.payload["source"] == "reference"  # анализируется эталон, а не ASR
    assert r.payload["text"] in ref
    assert len(r.payload["constructions"]) == 1  # выдуманная цитата отброшена
    assert any("отброшено цитат" in n for n in r.quality.notes)
    assert r.metric("valence").value == 0.4 and r.metric("valence").confidence is None
    assert len({x.payload["key"] for x in text}) == len(text)  # без повторной обработки одинакового текста
    tr = h.by("transactional")
    na = [x for x in tr if x.status.value == "not_applicable"]
    ok = [x for x in tr if x.status.value == "ok"]
    assert na and all(not has_direct_speech(x.payload["text"]) for x in na)  # повествование — без LLM
    assert ok and ok[0].payload["subject"] == "characters"
    reps = ok[0].payload["replicas"]
    assert reps[1]["label"] == "undetermined" and reps[1]["label_downgraded"]  # без обоснования метка снижается
    assert "не диагностика" in ok[0].payload["disclaimer"]
    # LLM вызывается только для текстовых единиц, не на каждом аудиочанке
    assert fake.chat_calls <= len(text) + len(ok)


async def test_qa_mode_answer_with_question(fixtures, monkeypatch):
    fake, _ = _setup(fixtures, monkeypatch)
    h = Harness(["speech", "text", "transactional"], mode="qa", question="Что вы думаете о встрече друзей?", overrides={**OV, "modules": {**OV["modules"], "text": {"min_chars": 40}}})
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=30)
    text = [r for r in h.by("text") if r.status.value == "ok"]
    assert text and text[0].payload["source"] == "participant_answer"
    assert any("Вопрос: «Что вы думаете о встрече друзей?»" in p for p in fake.prompts)
    tr = [r for r in h.by("transactional") if r.status.value == "ok"]
    assert tr and tr[0].payload["subject"] == "participant"


async def test_invalid_json_retry_and_errors(fixtures, monkeypatch):
    fake, ref = _setup(fixtures, monkeypatch, mode="invalid_then_ok")
    h = Harness(["speech", "text"], mode="reading", reference_text=ref, overrides=OV)
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=20)
    ok = [r for r in h.by("text") if r.status.value == "ok"]
    assert ok and ok[0].payload["llm"]["attempts"] == 2

    fake2, _ = _setup(fixtures, monkeypatch, mode="timeout")
    h2 = Harness(["speech", "text"], mode="reading", reference_text=ref, overrides=OV)
    await h2.run_file(fixtures / "reading_tts.wav", max_duration_s=20)
    errs = h2.by("text")
    assert errs and all(r.status.value == "error" and "таймаут" in r.error for r in errs)


async def test_without_llm_others_work(fixtures, monkeypatch):
    _setup(fixtures, monkeypatch, mode="down")
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    h = Harness(["speech", "text", "transactional"], mode="reading", reference_text=ref, overrides=OV)
    await h.run_file(fixtures / "reading_tts.wav", max_duration_s=20)
    st = {x.plugin.module_id: x.plugin.state.value for x in h.hosts}
    assert st["text"] == "unavailable" and st["transactional"] == "unavailable"
    assert "ollama" in [x for x in h.hosts if x.plugin.module_id == "text"][0].plugin.detail.lower()
    assert [r for r in h.by("speech") if r.status.value == "ok"]  # ASR работает без LLM
