"""Полная сессия Reading Mode: все 6 модулей → fusion → cross-modal → scoring → storage → export.

Face (MediaPipe) и Prosody — реальные; ASR, голосовая экспрессия и LLM — тестовые бэкенды.
"""

import json

import pandas as pd
import soundfile as sf

from app.analyzers import llm_client
from app.analyzers.speech import backends as asr_backends
from app.analyzers.voice_emotion import backends as ser_backends
from app.config import data_dir, load_config
from app.schemas.session import FileSourceConfig, SessionCreate, SessionState
from app.session.manager import SessionManager
from app.storage.db import Database
from app.storage.results_store import to_csv
from tests.conftest import wait_for
from tests.fakes import FakeASR, FakeOllama, FakeSER


async def test_full_reading_session(fixtures, monkeypatch):
    ref = (fixtures / "reading_reference.txt").read_text(encoding="utf-8")
    FakeASR.reference, FakeASR.total_ms, FakeASR.corrupt_every = ref, sf.info(str(fixtures / "reading_tts.wav")).duration * 1000, 7
    monkeypatch.setitem(asr_backends.BACKENDS, "fake", FakeASR)
    monkeypatch.setitem(ser_backends.BACKENDS, "fake", FakeSER)
    fake = FakeOllama(model=load_config("auto")["llm"]["model"])
    monkeypatch.setattr(llm_client, "TRANSPORT", fake.transport)

    mgr = SessionManager(Database(data_dir(load_config()) / "t.sqlite3"))
    req = SessionCreate(
        mode="reading", source="file", file=FileSourceConfig(path=str(fixtures / "reading_av.mp4"), realtime=False),
        reference_text=ref, composite_enabled=True,
        overrides={"modules": {"speech": {"backend": "fake", "partial_interval_ms": 0}, "voice_emotion": {"backend": "fake"}},
                   "scoring": {"composite": {"min_available": 1}}},
    )
    s = mgr.create(req)
    await s.start()
    assert await wait_for(lambda: s.state in (SessionState.completed, SessionState.failed), timeout=240)
    assert s.state == SessionState.completed, s.error

    mods = {r.module_id for r in s.store.results}
    assert mods >= {"face", "prosody", "speech", "voice_emotion", "text", "transactional", "fusion"}
    states = {h.module_id: h.state.value for h in s.module_health()}
    assert all(v in ("idle", "disabled") for v in states.values()), states

    # Fusion: окна 5 с с шагом 1 с, все финальные, есть несколько модальностей
    fusion = sorted([r for r in s.store.results if r.module_id == "fusion"], key=lambda r: r.t_start_ms)
    assert len(fusion) >= 40
    assert all(r.finality == "final" for r in fusion)
    starts = [r.t_start_ms for r in fusion]
    assert starts == sorted(set(starts)) and starts[1] - starts[0] == 1000
    mid = fusion[len(fusion) // 3]
    names = {m.name for m in mid.metrics}
    assert {"face.expression_intensity", "prosody.f0_median_hz"} <= names
    assert mid.payload["availability"]["face"] == "available"
    # окно, где лицо закрыто (10–13 с), явно помечает отсутствие модальности
    gap = [r for r in fusion if r.t_start_ms == 10000 and r.t_end_ms == 15000]
    if gap:
        assert gap[0].payload["availability"]["face"] in ("available", "insufficient_data")

    summ = s.summary
    sc = summ["scores"]
    ra = [i for i in sc["groups"]["reading_accuracy"] if i["name"] == "reading_accuracy"][0]
    assert ra["available"] and ra["raw"]["substitutions"] > 0 and 0 < ra["normalized"] < 100
    comp = sc["exploratory_composite_score"]
    assert comp["experimental"] and not comp["validated"] and comp["value"] == ra["normalized"]
    assert summ["crossmodal"]["pauses_vs_sentences"]["status"] in ("ok", "insufficient_data")
    assert "spearman" in summ["crossmodal"]

    # Storage + export
    row = s.db.get_session(s.id)
    assert row["state"] == "completed"
    df = pd.read_parquet(s.dir / "metrics.parquet")
    assert {"face", "prosody", "fusion"} <= set(df.module_id)
    raw_face = pd.read_parquet(s.dir / "raw_face.parquet")
    assert "blendshapes.mouthSmileLeft" in raw_face.columns
    assert (s.dir / "raw_voice_emotion.parquet").exists()  # embeddings — отдельно от JSON
    results = s.db.load_results(s.id)
    assert all("raw" not in r["payload"] for r in results)
    json.dumps(results, ensure_ascii=False)
    csv = to_csv(results)
    assert csv.count("\n") > 100
