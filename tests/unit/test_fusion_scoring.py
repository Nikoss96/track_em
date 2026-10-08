from app.config import load_config
from app.fusion.crossmodal import session_alignment
from app.fusion.engine import TemporalFusionEngine
from app.schemas.results import AnalysisResult, Metric, ResultStatus
from app.scoring.engine import RuleBasedScoring


def R(mod, t0, t1, **metrics):
    status = metrics.pop("_status", ResultStatus.ok)
    fin = metrics.pop("_finality", "final")
    payload = metrics.pop("_payload", {})
    ms = [Metric(name=k, value=v, method="t") for k, v in metrics.items()]
    return AnalysisResult(session_id="s", module_id=mod, model_id="m", model_version="v", t_start_ms=t0, t_end_ms=t1,
                          metrics=ms, status=status, finality=fin, payload=payload)


CFG = load_config("cpu")


def test_windows_overlap_weighting_and_availability():
    f = TemporalFusionEngine("s", CFG, ["face", "prosody", "text"])
    f.add(R("face", 0, 1000, smile=0.0))
    f.add(R("face", 1000, 5000, smile=1.0))
    out = f.tick(5000)
    assert len(out) == 1
    w = out[0]
    assert abs(w.metric("face.smile").value - 0.8) < 1e-9  # 1 с × 0 + 4 с × 1
    assert w.payload["availability"]["prosody"] == "missing"
    assert w.payload["availability"]["text"] == "pending"  # поздняя модальность ещё может прийти
    assert w.finality == "provisional"


def test_late_result_recomputes_window():
    f = TemporalFusionEngine("s", CFG, ["face", "text"])
    f.add(R("face", 0, 5000, smile=0.5))
    first = f.tick(6000)
    assert first[0].metric("text.valence") is None
    f.add(R("text", 1000, 4000, valence=0.7))  # поздний NLP-результат
    again = f.tick(6100)
    re = [r for r in again if r.t_start_ms == 0]
    assert re and re[0].metric("text.valence").value == 0.7
    assert f.recomputed >= 1
    final = f.finalize(7000)
    assert all(r.finality == "final" for r in final)


def test_provisional_replaced_by_final():
    f = TemporalFusionEngine("s", CFG, ["speech"])
    f.add(R("speech", 0, 3000, word_count=2, _finality="provisional", _payload={"replace_key": "seg1"}))
    f.add(R("speech", 0, 3000, word_count=5, _payload={"replace_key": "seg1"}))
    w = f.finalize(5000)[0]
    assert w.metric("speech.word_count").value == 5.0  # финальное значение заменило provisional
    assert len(f._by_module["speech"]) == 1


def test_scoring_reading_and_composite():
    rs = [R("speech", 0, 3000, _payload={"alignment": {"status": "ok", "substitutions": 1, "deletions": 1, "insertions": 0, "ref_words": 20}}),
          R("speech", 3000, 6000, _payload={"alignment": {"status": "unaligned"}}),
          R("face", 0, 1000, expression_intensity=0.2), R("face", 1000, 2000, expression_intensity=0.4)]
    sc = RuleBasedScoring().score(rs, {}, {**CFG, "composite_enabled": None})
    ra = sc["groups"]["reading_accuracy"][0]
    assert ra["raw"]["wer"] == 0.1 and ra["normalized"] == 90.0 and ra["raw"]["segments"] == 1
    fe = [i for i in sc["groups"]["facial_expressivity"] if i["name"] == "expression_intensity"][0]
    assert abs(fe["raw"] - 0.3) < 1e-9 and fe["normalized"] is None  # без процедуры нормализации — только raw
    assert sc["exploratory_composite_score"]["enabled"] is False and sc["exploratory_composite_score"]["value"] is None
    cfg2 = {**CFG, "composite_enabled": True}
    c = RuleBasedScoring().score(rs, {}, cfg2)["exploratory_composite_score"]
    assert c["value"] is None and "недостаточно" in c["reason"]  # min_available=2, есть только 1
    cfg3 = {**cfg2, "scoring": {**CFG["scoring"], "composite": {"min_available": 1, "weights": {"reading_accuracy": 1}}}}
    c3 = RuleBasedScoring().score(rs, {}, cfg3)["exploratory_composite_score"]
    assert c3["value"] == 90.0 and c3["experimental"] and not c3["validated"]


def test_session_alignment_descriptive():
    wins = [{"face.expression_intensity": i / 10, "prosody.f0_sd_st": i / 5 + (0.1 if i % 2 else 0)} for i in range(12)]
    out = session_alignment(wins, [(900, 1500), (5000, 5600)], [1200, 3000, 5300])
    assert out["spearman"]["face_vs_voice"]["n"] == 12 and out["spearman"]["face_vs_voice"]["rho"] > 0.9
    assert out["spearman"]["text_vs_voice"]["status"] == "insufficient_data"
    p = out["pauses_vs_sentences"]
    assert p["pauses_at_boundaries_ratio"] == 1.0 and abs(p["boundaries_with_pause_ratio"] - 2 / 3) < 1e-3
