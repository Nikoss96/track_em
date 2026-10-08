"""ScoringEngine: групповые сводки сессии.

Правила:
- raw-значения хранятся всегда;
- normalized (0–100) — только по процедуре, явно заданной в configs (scoring.normalizations);
- exploratory_composite_score — выключен по умолчанию, только по доступным нормализованным
  метрикам с весами из YAML, иначе null; всегда помечен как экспериментальный.

Интерфейс score(...) позволяет заменить правила обученной fusion-моделью (см. ScoringBackend).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

import numpy as np

from app.schemas.results import AnalysisResult

GROUPS = (
    "facial_expressivity",
    "prosodic_expressivity",
    "voice_emotion_profile",
    "reading_accuracy",
    "text_sentiment",
    "transactional_style",
    "crossmodal_consistency",
)


class ScoringBackend(Protocol):
    def score(self, results: list[AnalysisResult], crossmodal: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]: ...


def _vals(results: list[AnalysisResult], module: str, metric: str) -> list[float]:
    out = []
    for r in results:
        if r.module_id != module or r.payload.get("summary") or r.finality != "final":
            continue
        m = r.metric(metric)
        if m is not None and m.status.value in ("ok", "partial") and isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
            if np.isfinite(m.value):
                out.append(float(m.value))
    return out


def _labels(results: list[AnalysisResult], module: str, metric: str) -> list[str]:
    out = []
    for r in results:
        if r.module_id != module or r.finality != "final":
            continue
        m = r.metric(metric)
        if m is not None and isinstance(m.value, str) and m.status.value in ("ok", "partial"):
            out.append(m.value)
    return out


def _stat(name: str, vals: list[float], unit: str | None, method: str) -> dict[str, Any]:
    if not vals:
        return {"name": name, "raw": None, "unit": unit, "available": False, "n": 0, "method": method}
    return {
        "name": name,
        "raw": round(float(np.mean(vals)), 5),
        "sd": round(float(np.std(vals)), 5) if len(vals) > 1 else None,
        "median": round(float(np.median(vals)), 5),
        "unit": unit,
        "available": True,
        "n": len(vals),
        "method": method,
    }


def _dist(name: str, labels: list[str]) -> dict[str, Any]:
    if not labels:
        return {"name": name, "raw": None, "available": False, "n": 0, "method": "label distribution"}
    c = Counter(labels)
    n = len(labels)
    return {"name": name, "raw": {k: round(v / n, 4) for k, v in c.most_common()}, "available": True, "n": n, "method": "label distribution (доли)"}


class RuleBasedScoring:
    version = "0.1.0"

    def score(self, results: list[AnalysisResult], crossmodal: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
        sc = cfg["scoring"]
        norms = sc.get("normalizations", {}) or {}
        groups: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}

        groups["facial_expressivity"] += [
            _stat("expression_intensity", _vals(results, "face", "expression_intensity"), "coef", "среднее по окнам 1 с (blendshapes)"),
            _stat("expression_variability", _vals(results, "face", "expression_variability"), "coef", "SD интенсивности внутри окна"),
            _stat("smile", _vals(results, "face", "smile"), "coef", "mouthSmileL/R"),
            _stat("brow_raise", _vals(results, "face", "brow_raise"), "coef", "browInnerUp/browOuterUp"),
            _stat("face_presence", _vals(results, "face", "face_presence"), "ratio", "доля кадров с лицом"),
        ]
        groups["prosodic_expressivity"] += [
            _stat("f0_median_hz", _vals(results, "prosody", "f0_median_hz"), "Hz", "Praat autocorrelation, озвученные кадры"),
            _stat("f0_sd_st", _vals(results, "prosody", "f0_sd_st"), "st", "SD F0 в полутонах (контекстное окно)"),
            _stat("f0_range_st", _vals(results, "prosody", "f0_range_st"), "st", "P90–P10 F0 в полутонах"),
            _stat("intensity_sd_db", _vals(results, "prosody", "intensity_sd_db"), "dB", "SD RMS-энергии"),
            _stat("pause_ratio", _vals(results, "prosody", "pause_ratio"), "ratio", "доля пауз в окне"),
            _stat("speech_rate_wpm", _vals(results, "prosody", "speech_rate_wpm"), "words/min", "по финальной транскрипции"),
        ]
        groups["voice_emotion_profile"] += [
            _dist("top_label", _labels(results, "voice_emotion", "top_label")),
            _stat("positive_minus_negative", _vals(results, "voice_emotion", "positive_minus_negative"), "score", "happy − (angry+sad+fearful+disgusted), выход модели"),
        ]
        # reading accuracy — нормализация по явной формуле ТЗ
        ra = self._reading(results, norms.get("reading_accuracy"))
        groups["reading_accuracy"].append(ra)
        groups["text_sentiment"] += [
            _dist("sentiment", _labels(results, "text", "sentiment")),
            _stat("valence", _vals(results, "text", "valence"), "[-1,1]", "LLM, оценка текста (не чтеца)"),
            _stat("emotional_intensity", _vals(results, "text", "emotional_intensity"), "[0,1]", "LLM, оценка текста"),
        ]
        groups["transactional_style"].append(_dist("primary_label", _labels(results, "transactional", "primary_label")))
        groups["crossmodal_consistency"].append(
            {"name": "descriptive_alignment", "raw": crossmodal, "available": bool(crossmodal), "method": "см. docs/metrics.md#cross-modal"}
        )
        # линейные нормализации, явно заданные в YAML: {metric: {method: linear, min, max, source}}
        for g, items in groups.items():
            for it in items:
                spec = norms.get(f"{g}.{it['name']}")
                if spec and spec.get("method") == "linear" and isinstance(it.get("raw"), (int, float)):
                    lo, hi = float(spec["min"]), float(spec["max"])
                    it["normalized"] = round(float(np.clip(100 * (it["raw"] - lo) / (hi - lo), 0, 100)), 2)
                    it["normalization"] = {"method": "linear", "min": lo, "max": hi, "source": spec.get("source")}
                it.setdefault("normalized", None)
                it.setdefault("normalization_version", sc.get("normalization_version"))

        composite = self._composite(groups, sc.get("composite", {}), cfg.get("composite_enabled"))
        return {"version": self.version, "normalization_version": sc.get("normalization_version"), "groups": groups, "exploratory_composite_score": composite}

    @staticmethod
    def _reading(results: list[AnalysisResult], spec: dict[str, Any] | None) -> dict[str, Any]:
        S = D = I = N = 0
        segs = 0
        for r in results:
            if r.module_id != "speech" or r.finality != "final":
                continue
            al = r.payload.get("alignment")
            if not al or al.get("status") != "ok":
                continue
            S += al["substitutions"]
            D += al["deletions"]
            I += al["insertions"]
            N += al["ref_words"]
            segs += 1
        if N == 0:
            return {"name": "reading_accuracy", "raw": None, "available": False, "method": "WER на выровненных завершённых фрагментах", "normalized": None}
        wer = (S + D + I) / N
        return {
            "name": "reading_accuracy",
            "raw": {"wer": round(wer, 4), "substitutions": S, "deletions": D, "insertions": I, "ref_words": N, "segments": segs},
            "available": True,
            "method": "WER на выровненных завершённых фрагментах",
            "normalized": round(100 * max(0.0, 1 - wer), 2),
            "normalization": {"method": (spec or {}).get("method", "100*max(0,1-WER)"), "source": (spec or {}).get("source", "ТЗ §6")},
        }

    @staticmethod
    def _composite(groups: dict[str, list[dict[str, Any]]], comp: dict[str, Any], session_flag: Any) -> dict[str, Any]:
        enabled = bool(session_flag) if session_flag is not None else bool(comp.get("enabled"))
        base = {
            "enabled": enabled,
            "experimental": True,
            "validated": False,
            "label": "Экспериментальный невалидированный индекс. Не является оценкой личности, интеллекта, честности или психического состояния.",
            "value": None,
            "weights": comp.get("weights", {}),
            "used": [],
        }
        if not enabled:
            return base
        weights: dict[str, float] = comp.get("weights", {}) or {}
        used, num, den = [], 0.0, 0.0
        for g, items in groups.items():
            for it in items:
                key = f"{g}.{it['name']}" if f"{g}.{it['name']}" in weights else it["name"] if it["name"] in weights else None
                if key and isinstance(it.get("normalized"), (int, float)):
                    w = float(weights[key])
                    num += w * it["normalized"]
                    den += w
                    used.append({"metric": key, "normalized": it["normalized"], "weight": w})
        base["used"] = used
        if den > 0 and len(used) >= int(comp.get("min_available", 2)):
            base["value"] = round(num / den, 2)
        else:
            base["reason"] = f"недостаточно нормализованных метрик ({len(used)} < {comp.get('min_available', 2)})"
        return base
