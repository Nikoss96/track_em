"""Module B: ProsodyAnalyzer — объективные акустические измерения.

- instantaneous (окно window_ms, шаг hop_ms): F0 по озвученным кадрам, voiced_ratio, RMS;
- window-level (окно context_window_ms, шаг context/2): F0 median/SD/range в полутонах,
  динамика интенсивности, паузы, ритм, eGeMAPS (openSMILE);
- speech rate — по transcript.final (слов/мин, артикуляционная скорость без пауз).
F0 не вычисляется по неозвученным участкам; тишина/шум/короткие фрагменты → insufficient_data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.analyzers.base import AnalyzerPlugin
from app.analyzers.prosody import features as F
from app.schemas.events import AudioChunk, Event, EventType, Transcript
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus

M = ResultStatus


class ProsodyAnalyzer(AnalyzerPlugin):
    module_id = "prosody"
    version = "0.1.0"
    subscribes = (EventType.audio_chunk, EventType.transcript_final)
    queue_policy = "lossless"
    queue_size = 2000

    async def initialize(self, config: dict[str, Any]) -> None:
        import librosa
        import opensmile
        import parselmouth

        self.cfg = config
        self.win = int(config.get("window_ms", 1000))
        self.hop = int(config.get("hop_ms", 500))
        self.ctx_win = int(config.get("context_window_ms", 4000))
        self.ctx_hop = max(self.ctx_win // 2, self.hop)
        self.floor = float(config.get("f0_min_hz", 75))
        self.ceil = float(config.get("f0_max_hz", 500))
        self.use_egemaps = bool(config.get("egemaps", True))
        self.silence_dbfs = float(self.ctx.config["capture"]["silence_dbfs"])
        self.model_id = "praat-ac + opensmile-eGeMAPSv02"
        self.model_version = f"parselmouth {parselmouth.__version__}; opensmile {opensmile.__version__}; librosa {librosa.__version__}"
        self.pauses = F.PauseDetector(self.silence_dbfs, float(config.get("pause_min_ms", 250)))
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0: float | None = None
        self.next_inst: float | None = None
        self.next_ctx: float | None = None
        self._reported_pauses = 0

    # --- буфер ------------------------------------------------------------
    def _append(self, ch: AudioChunk) -> None:
        if self.buf_t0 is None or ch.gap_before_ms > 0:
            self.buf = ch.pcm16k.copy()
            self.buf_t0 = float(ch.t_ms)
            self.next_inst = self.buf_t0 + self.win
            self.next_ctx = self.buf_t0 + self.ctx_win
            self.pauses.reset()
        else:
            self.buf = np.concatenate([self.buf, ch.pcm16k])
        keep = int((self.ctx_win + 1000) * 16)
        if self.buf.size > keep:
            drop = self.buf.size - keep
            self.buf = self.buf[drop:]
            self.buf_t0 += drop / 16.0

    def _slice(self, t0: float, t1: float) -> np.ndarray:
        assert self.buf_t0 is not None
        a = int(round((t0 - self.buf_t0) * 16))
        b = int(round((t1 - self.buf_t0) * 16))
        return self.buf[max(a, 0) : max(b, 0)]

    @property
    def buf_end(self) -> float:
        return (self.buf_t0 or 0.0) + self.buf.size / 16.0

    # --- обработка --------------------------------------------------------
    async def process(self, event: Event) -> list[AnalysisResult]:
        if isinstance(event, Transcript):
            return self._speech_rate(event)
        assert isinstance(event, AudioChunk)
        self._append(event)
        self.pauses.feed(F.frame_db(event.pcm16k), float(event.t_ms))
        out: list[AnalysisResult] = []
        while self.next_inst is not None and self.buf_end >= self.next_inst:
            t1 = self.next_inst
            x = self._slice(t1 - self.win, t1)
            out.append(await self.run_blocking(self._instantaneous, x, t1 - self.win, t1, gated=False))
            self.next_inst += self.hop
        while self.next_ctx is not None and self.buf_end >= self.next_ctx:
            t1 = self.next_ctx
            x = self._slice(t1 - self.ctx_win, t1)
            out.append(await self.run_blocking(self._context, x, t1 - self.ctx_win, t1, gated=False))
            self.next_ctx += self.ctx_hop
        return out

    def _instantaneous(self, x: np.ndarray, t0: float, t1: float) -> AnalysisResult:
        rms = float(F.frame_db(x).mean()) if x.size >= F.FRAME else -120.0
        level = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9) if x.size else -120.0
        metrics = [Metric(name="rms_dbfs", value=round(float(level), 2), unit="dBFS", method="RMS окна", group="instantaneous")]
        q = Quality()
        if level < self.silence_dbfs:
            q.notes.append("тишина")
            return self.result(t0, t1, metrics, M.insufficient_data, quality=q, payload={"scope": "instantaneous"})
        f0 = F.f0_contour(x, self.floor, self.ceil)
        st = F.f0_stats(f0, min_voiced=5)
        metrics.append(Metric(name="voiced_ratio", value=round(st["voiced_ratio"], 4), unit="ratio", method="доля озвученных 10-мс кадров (Praat)", group="instantaneous"))
        if st["ok"]:
            metrics.append(Metric(name="f0_mean_hz", value=round(st["f0_mean_hz"], 2), unit="Hz", method="Praat AC, только озвученные кадры", group="instantaneous"))
            status = M.ok
        else:
            metrics.append(Metric(name="f0_mean_hz", value=None, unit="Hz", method="Praat AC", status=M.insufficient_data, group="instantaneous"))
            q.notes.append("мало озвученных кадров")
            status = M.partial
        metrics.append(Metric(name="frame_db_mean", value=round(rms, 2), unit="dBFS", method="среднее 10-мс энергии", group="instantaneous"))
        return self.result(t0, t1, metrics, status, quality=q, payload={"scope": "instantaneous", "f0_contour": [round(v, 1) for v in f0.tolist()]})

    def _context(self, x: np.ndarray, t0: float, t1: float) -> AnalysisResult:
        q = Quality()
        db = F.frame_db(x)
        level = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)) if x.size else -120.0
        thr = max(self.silence_dbfs, float(np.percentile(db, 95)) - 25.0) if db.size else self.silence_dbfs
        speech = db >= thr
        speech_ms = float(speech.sum() * 10)
        metrics: list[Metric] = [
            Metric(name="rms_dbfs_window", value=round(level, 2), unit="dBFS", method="RMS окна", group="window"),
            Metric(name="speech_ms", value=round(speech_ms), unit="ms", method="10-мс кадры выше порога энергии", group="window"),
        ]
        # паузы, перекрывающие окно (обрезанные границами окна)
        pa = [(max(a, t0), min(b, t1)) for a, b in self.pauses.pauses if b > t0 and a < t1]
        pause_total = sum(b - a for a, b in pa)
        n_p = sum(1 for a, b in self.pauses.pauses if t0 <= (a + b) / 2 < t1)
        metrics += [
            Metric(name="pause_count", value=n_p, unit="count", method=f"тишина ≥ {self.pauses.min_ms:.0f} мс между речью", group="window"),
            Metric(name="pause_total_ms", value=round(pause_total), unit="ms", method="сумма пауз в окне", group="window"),
            Metric(name="pause_ratio", value=round(pause_total / (t1 - t0), 4), unit="ratio", method="доля пауз в окне", group="window"),
        ]
        if pa:
            metrics.append(Metric(name="pause_mean_ms", value=round(pause_total / len(pa)), unit="ms", method="средняя длительность паузы", group="window"))
        new_pauses = self.pauses.pauses[self._reported_pauses :]
        self._reported_pauses = len(self.pauses.pauses)
        payload: dict[str, Any] = {"scope": "window", "pauses": [[round(a), round(b)] for a, b in new_pauses]}
        if level < self.silence_dbfs or speech_ms < 300:
            q.notes.append("тишина или слишком мало речи")
            return self.result(t0, t1, metrics, M.insufficient_data, quality=q, payload=payload)
        sp_db = db[speech]
        metrics += [
            Metric(name="intensity_mean_db", value=round(float(sp_db.mean()), 2), unit="dBFS", method="среднее 10-мс энергии речевых кадров", group="window"),
            Metric(name="intensity_sd_db", value=round(float(sp_db.std()), 3), unit="dB", method="SD 10-мс энергии речевых кадров", group="window"),
            Metric(name="intensity_range_db", value=round(float(np.percentile(sp_db, 95) - np.percentile(sp_db, 5)), 2), unit="dB", method="P95–P5 энергии речи", group="window"),
        ]
        f0 = F.f0_contour(x, self.floor, self.ceil)
        st = F.f0_stats(f0, min_voiced=20)
        status = M.ok
        if st["ok"]:
            for k, unit, meth in (
                ("f0_median_hz", "Hz", "медиана F0 озвученных кадров"),
                ("f0_sd_st", "st", "SD F0 в полутонах относительно медианы"),
                ("f0_range_st", "st", "P90–P10 F0 в полутонах"),
                ("f0_min_hz", "Hz", "P5 F0"),
                ("f0_max_hz", "Hz", "P95 F0"),
            ):
                metrics.append(Metric(name=k, value=round(st[k], 3), unit=unit, method=f"Praat AC; {meth}", group="window"))
        else:
            q.notes.append("недостаточно озвученных кадров для F0")
            status = M.partial
        metrics.append(Metric(name="voiced_ratio_window", value=round(st["voiced_ratio"], 4), unit="ratio", method="доля озвученных кадров", group="window"))
        for k, v in F.rhythm_stats(f0, db).items():
            unit = {"voiced_segment_varco": "%", "voiced_segment_mean_ms": "ms", "syllable_nuclei_rate": "1/s"}[k]
            metrics.append(Metric(name=k, value=round(v, 3), unit=unit, method="ритм: озвученные участки / пики интенсивности", group="rhythm"))
        if self.use_egemaps:
            try:
                eg = F.egemaps(x)
                payload["egemaps"] = eg
                for src, (name, unit) in F.EGEMAPS_KEYS.items():
                    if eg.get(src) is not None:
                        metrics.append(Metric(name=name, value=round(eg[src], 5), unit=unit, method=f"openSMILE eGeMAPSv02: {src}", group="egemaps"))
            except Exception as e:  # noqa: BLE001
                q.notes.append(f"eGeMAPS: {e}")
        return self.result(t0, t1, metrics, status, quality=q, payload=payload)

    def _speech_rate(self, tr: Transcript) -> list[AnalysisResult]:
        words = [w for w in tr.text.split() if any(c.isalnum() for c in w)]
        dur = tr.t_end_ms - tr.t_ms
        if dur < 500 or not words:
            return [self.result(tr.t_ms, tr.t_end_ms, [Metric(name="speech_rate_wpm", value=None, unit="words/min", method="ASR", status=M.insufficient_data, group="rate")], M.insufficient_data, payload={"scope": "rate"})]
        pause_ms = sum(min(b, tr.t_end_ms) - max(a, tr.t_ms) for a, b in self.pauses.pauses if b > tr.t_ms and a < tr.t_end_ms)
        art = dur - pause_ms
        metrics = [
            Metric(name="speech_rate_wpm", value=round(len(words) / dur * 60000, 1), unit="words/min", method="слов финальной транскрипции / длительность фразы", group="rate"),
            Metric(name="word_count", value=len(words), unit="words", method="ASR", group="rate"),
        ]
        if art > 300:
            metrics.append(Metric(name="articulation_rate_wpm", value=round(len(words) / art * 60000, 1), unit="words/min", method="слов / (длительность − паузы)", group="rate"))
        return [self.result(tr.t_ms, tr.t_end_ms, metrics, payload={"scope": "rate", "segment_id": tr.segment_id})]

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        new = self.pauses.pauses[self._reported_pauses :]
        if not new:
            return []
        self._reported_pauses = len(self.pauses.pauses)
        a, b = new[0][0], new[-1][1]
        return [self.result(a, b, [], M.ok, payload={"scope": "pauses", "summary": True, "pauses": [[round(x), round(y)] for x, y in new]})]
