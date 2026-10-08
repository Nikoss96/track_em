"""Акустические измерения (чистые функции): F0 (Praat), интенсивность, паузы, ритм, eGeMAPS."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SR = 16000
FRAME = 160  # 10 мс


def frame_db(x: np.ndarray) -> np.ndarray:
    n = x.size // FRAME
    if n == 0:
        return np.zeros(0)
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-9)


def f0_contour(x: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    """F0 (Гц) с шагом 10 мс; 0 — неозвученный кадр. Praat autocorrelation."""
    import parselmouth

    if x.size < int(SR * 3 / floor) + FRAME:
        return np.zeros(0)
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    return np.asarray(pitch.selected_array["frequency"], dtype=np.float64)


def semitones(f0: np.ndarray, ref: float) -> np.ndarray:
    return 12 * np.log2(f0 / ref)


def f0_stats(f0: np.ndarray, min_voiced: int = 10) -> dict[str, Any]:
    voiced = f0[f0 > 0]
    out: dict[str, Any] = {"voiced_frames": int(voiced.size), "voiced_ratio": float(voiced.size / f0.size) if f0.size else 0.0}
    if voiced.size < min_voiced:
        out["ok"] = False
        return out
    med = float(np.median(voiced))
    st = semitones(voiced, med)
    out.update(
        ok=True,
        f0_mean_hz=float(voiced.mean()),
        f0_median_hz=med,
        f0_sd_st=float(st.std()),
        f0_range_st=float(np.percentile(st, 90) - np.percentile(st, 10)),
        f0_min_hz=float(np.percentile(voiced, 5)),
        f0_max_hz=float(np.percentile(voiced, 95)),
    )
    return out


def voiced_segments(f0: np.ndarray, step_ms: float = 10.0) -> list[float]:
    """Длительности (мс) непрерывных озвученных участков."""
    durs, run = [], 0
    for v in f0 > 0:
        if v:
            run += 1
        elif run:
            durs.append(run * step_ms)
            run = 0
    if run:
        durs.append(run * step_ms)
    return durs


def rhythm_stats(f0: np.ndarray, db: np.ndarray) -> dict[str, Any]:
    """Ритм: вариативность длительностей озвученных участков (Varco) и частота пиков
    интенсивности в озвученных участках (прокси слоговых ядер, по мотивам De Jong & Wempe, 2009)."""
    from scipy.signal import find_peaks

    out: dict[str, Any] = {}
    segs = [d for d in voiced_segments(f0) if d >= 30]
    if len(segs) >= 3:
        out["voiced_segment_varco"] = float(100 * np.std(segs) / np.mean(segs))
        out["voiced_segment_mean_ms"] = float(np.mean(segs))
    n = min(f0.size, db.size)
    if n > 20:
        d = np.convolve(db[:n], np.ones(3) / 3, mode="same")
        peaks, _ = find_peaks(d, prominence=2.0, distance=5)
        peaks = [p for p in peaks if f0[p] > 0]
        voiced_s = float((f0[:n] > 0).sum()) / 100.0
        if voiced_s >= 0.5:
            out["syllable_nuclei_rate"] = float(len(peaks) / voiced_s)
    return out


class PauseDetector:
    """Потоковое выделение пауз по 10-мс энергии: тишина ≥ min_ms между участками речи."""

    def __init__(self, silence_dbfs: float, min_ms: float, rel_db: float = 25.0) -> None:
        self.abs_thr = silence_dbfs
        self.rel_db = rel_db
        self.min_ms = min_ms
        self.hist: list[float] = []
        self.had_speech = False
        self.sil_start: float | None = None
        self.pauses: list[tuple[float, float]] = []

    def feed(self, db: np.ndarray, t0_ms: float) -> list[tuple[float, float]]:
        new = []
        self.hist.extend(db.tolist())
        self.hist = self.hist[-500:]
        ref = float(np.percentile(self.hist, 95)) if self.hist else -20.0
        thr = max(self.abs_thr, ref - self.rel_db)
        for i, v in enumerate(db):
            t = t0_ms + i * 10.0
            if v < thr:
                if self.sil_start is None:
                    self.sil_start = t
            else:
                if self.sil_start is not None and self.had_speech and t - self.sil_start >= self.min_ms:
                    p = (self.sil_start, t)
                    self.pauses.append(p)
                    new.append(p)
                self.sil_start = None
                self.had_speech = True
        return new

    def reset(self) -> None:
        self.sil_start = None
        self.had_speech = False


_SMILE: Any = None
EGEMAPS_KEYS = {
    "loudness_sma3_amean": ("egemaps_loudness_mean", "sone"),
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": ("egemaps_f0_cv", "ratio"),
    "jitterLocal_sma3nz_amean": ("egemaps_jitter_local", "ratio"),
    "shimmerLocaldB_sma3nz_amean": ("egemaps_shimmer_db", "dB"),
    "HNRdBACF_sma3nz_amean": ("egemaps_hnr_db", "dB"),
    "VoicedSegmentsPerSec": ("egemaps_voiced_segments_per_s", "1/s"),
    "MeanUnvoicedSegmentLength": ("egemaps_mean_unvoiced_s", "s"),
}


def egemaps(x: np.ndarray) -> dict[str, float]:
    global _SMILE
    import opensmile

    if _SMILE is None:
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    df = _SMILE.process_signal(x.astype(np.float32), SR)
    row = df.iloc[0]
    return {k: (float(v) if math.isfinite(float(v)) else None) for k, v in row.items()}
