"""Cross-Modal Alignment.

Принцип: сохраняем сопоставленные признаки и расхождения; не вводим «оценку соответствия»
без валидированной шкалы. Расхождение модальностей ≠ ошибка исполнения.

- window_alignment(): сопоставление признаков в одном окне (без интегральной оценки);
- session_alignment(): описательная статистика по сессии — ранговые корреляции Спирмена
  между рядами окон, сравнение эмоционально значимых фрагментов (Cohen's d) и доля пауз,
  совпадающих с границами предложений.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Пары рядов для описательных корреляций (модуль.метрика).
PAIRS: list[tuple[str, str, str]] = [
    ("face_vs_voice", "face.expression_intensity", "prosody.f0_sd_st"),
    ("face_vs_voice_energy", "face.expression_intensity", "prosody.intensity_sd_db"),
    ("text_vs_voice", "text.valence", "voice_emotion.positive_minus_negative"),
    ("text_vs_face", "text.valence", "face.smile"),
    ("text_intensity_vs_face", "text.emotional_intensity", "face.expression_intensity"),
    ("text_intensity_vs_voice", "text.emotional_intensity", "prosody.f0_sd_st"),
]

FACE_KEYS = ("face.expression_intensity", "face.smile", "face.brow_raise", "face.face_presence")
VOICE_KEYS = ("prosody.f0_median_hz", "prosody.f0_sd_st", "prosody.intensity_sd_db", "voice_emotion.top_label", "voice_emotion.positive_minus_negative")
TEXT_KEYS = ("text.sentiment", "text.valence", "text.emotional_intensity")


def window_alignment(fused: dict[str, Any]) -> dict[str, Any]:
    pick = lambda keys: {k: fused[k] for k in keys if k in fused}  # noqa: E731
    face, voice, text = pick(FACE_KEYS), pick(VOICE_KEYS), pick(TEXT_KEYS)
    out: dict[str, Any] = {"face": face, "voice": voice, "text": text}
    # Простое описательное расхождение знаков (только если обе величины определены).
    tv, vv = fused.get("text.valence"), fused.get("voice_emotion.positive_minus_negative")
    if isinstance(tv, (int, float)) and isinstance(vv, (int, float)):
        out["text_voice_sign"] = "same" if (tv >= 0) == (vv >= 0) else "different"
    return out


def _spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    n = len(x)
    if n < 8:
        return {"rho": None, "p_value": None, "n": n, "status": "insufficient_data"}
    from scipy.stats import spearmanr

    xs, ys = np.asarray(x), np.asarray(y)
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return {"rho": None, "p_value": None, "n": n, "status": "insufficient_data", "note": "constant series"}
    rho, p = spearmanr(xs, ys)
    return {"rho": round(float(rho), 4), "p_value": round(float(p), 6), "n": n, "status": "ok"}


def _cohens_d(a: list[float], b: list[float]) -> dict[str, Any]:
    if len(a) < 3 or len(b) < 3:
        return {"d": None, "n_a": len(a), "n_b": len(b), "status": "insufficient_data"}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if sp == 0:
        return {"d": None, "n_a": len(a), "n_b": len(b), "status": "insufficient_data"}
    return {"d": round((ma - mb) / sp, 4), "mean_a": round(ma, 4), "mean_b": round(mb, 4), "n_a": len(a), "n_b": len(b), "status": "ok"}


def session_alignment(
    windows: list[dict[str, Any]],
    pauses: list[tuple[int, int]],
    sentence_boundaries_ms: list[int],
    emotional_threshold: float = 0.6,
    boundary_tolerance_ms: int = 400,
) -> dict[str, Any]:
    """windows — список {metric_name: value} по окнам fusion (без перекрытия логика не меняется)."""
    out: dict[str, Any] = {"method_note": "Описательная статистика; не является валидированной шкалой согласованности."}
    corr = {}
    for name, a, b in PAIRS:
        xs, ys = [], []
        for w in windows:
            va, vb = w.get(a), w.get(b)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                xs.append(float(va))
                ys.append(float(vb))
        corr[name] = {"x": a, "y": b, **_spearman(xs, ys)}
    out["spearman"] = corr
    # 4. Экспрессия в эмоционально значимых фрагментах vs остальные
    sig: dict[str, Any] = {}
    for feat in ("face.expression_intensity", "prosody.f0_sd_st", "prosody.intensity_sd_db"):
        hi, lo = [], []
        for w in windows:
            ti, fv = w.get("text.emotional_intensity"), w.get(feat)
            if isinstance(ti, (int, float)) and isinstance(fv, (int, float)):
                (hi if ti >= emotional_threshold else lo).append(float(fv))
        sig[feat] = _cohens_d(hi, lo)
    out["emotional_fragments"] = {"threshold": emotional_threshold, "effect_sizes": sig}
    # 5. Паузы vs границы предложений
    if pauses and sentence_boundaries_ms:
        b = np.asarray(sorted(sentence_boundaries_ms))
        hits = 0
        for ps, pe in pauses:
            mid = (ps + pe) / 2
            j = np.searchsorted(b, mid)
            near = [abs(b[k] - mid) for k in (j - 1, j) if 0 <= k < len(b)]
            if near and min(near) <= max(boundary_tolerance_ms, (pe - ps) / 2 + boundary_tolerance_ms):
                hits += 1
        covered = 0
        for t in b:
            if any(ps - boundary_tolerance_ms <= t <= pe + boundary_tolerance_ms for ps, pe in pauses):
                covered += 1
        out["pauses_vs_sentences"] = {
            "pauses": len(pauses),
            "sentence_boundaries": len(b),
            "pauses_at_boundaries_ratio": round(hits / len(pauses), 4),
            "boundaries_with_pause_ratio": round(covered / len(b), 4),
            "tolerance_ms": boundary_tolerance_ms,
            "status": "ok",
        }
    else:
        out["pauses_vs_sentences"] = {"status": "insufficient_data", "pauses": len(pauses), "sentence_boundaries": len(sentence_boundaries_ms)}
    return out
