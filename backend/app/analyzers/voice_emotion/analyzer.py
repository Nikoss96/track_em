"""Module C: VoiceEmotionAnalyzer — голосовая эмоциональная экспрессия по речевым сегментам VAD.

Окна window_ms (шаг hop_ms) внутри завершённого речевого сегмента; сегменты короче
min_speech_ms → insufficient_data. Классификационный результат (scores модели) и сырой
embedding сохраняются раздельно. Результаты — выход модели, а не установленное состояние человека.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.analyzers.base import AnalyzerPlugin, ModelUnavailable
from app.analyzers.voice_emotion.backends import BACKENDS
from app.config import ROOT
from app.schemas.events import AudioChunk, Event, EventType, SpeechActivity
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus

M = ResultStatus
NEGATIVE = ("angry", "sad", "fearful", "disgusted")
VALIDATION_NOTE = "not_validated_ru: применимость весов к русской речи не подтверждена валидацией"


def windows(start: float, end: float, win: float, hop: float) -> list[tuple[float, float]]:
    if end - start <= win:
        return [(start, end)]
    out, t = [], start
    while t + win <= end:
        out.append((t, t + win))
        t += hop
    if out[-1][1] < end:  # хвост сегмента покрывается окном, выровненным по концу
        out.append((end - win, end))
    return out


class VoiceEmotionAnalyzer(AnalyzerPlugin):
    module_id = "voice_emotion"
    version = "0.1.0"
    subscribes = (EventType.audio_chunk, EventType.speech_activity)
    queue_policy = "lossless"
    queue_size = 3000

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        name = config.get("backend", "emotion2vec")
        if name not in BACKENDS:
            raise ModelUnavailable(f"неизвестный backend: {name}")
        self.backend = BACKENDS[name](config, str(ROOT / ".cache"))
        await self.run_blocking(self.backend.load, gated=True)
        self.model_id = self.backend.model_id
        self.model_version = self.backend.model_version
        self.ctx.shared["register_model"](**self.backend.registry)
        self.win = float(config.get("window_ms", 4000))
        self.hop = float(config.get("hop_ms", 2000))
        self.min_speech = float(config.get("min_speech_ms", 1500))
        self.save_emb = bool(config.get("save_embedding", True))
        self.silence_dbfs = float(self.ctx.config["capture"]["silence_dbfs"])
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0: float | None = None

    def _append(self, ch: AudioChunk) -> None:
        if self.buf_t0 is None or ch.gap_before_ms > 0:
            self.buf, self.buf_t0 = ch.pcm16k.copy(), float(ch.t_ms)
        else:
            self.buf = np.concatenate([self.buf, ch.pcm16k])
        keep = 40 * 16000
        if self.buf.size > keep:
            self.buf_t0 += (self.buf.size - keep) / 16.0
            self.buf = self.buf[-keep:]

    def _slice(self, t0: float, t1: float) -> np.ndarray:
        assert self.buf_t0 is not None
        a = max(int((t0 - self.buf_t0) * 16), 0)
        return self.buf[a : max(a, int((t1 - self.buf_t0) * 16))]

    async def process(self, event: Event) -> list[AnalysisResult]:
        if isinstance(event, AudioChunk):
            self._append(event)
            return []
        assert isinstance(event, SpeechActivity)
        if event.kind != "end" or self.buf_t0 is None:
            return []
        s, e = float(event.t_ms), float(event.t_end_ms)
        if e - s < self.min_speech:
            return [self.result(s, e, [], M.insufficient_data, quality=Quality(notes=[f"речевой сегмент {e - s:.0f} мс < {self.min_speech:.0f} мс"]),
                                payload={"segment_id": event.segment_id})]
        out = []
        for a, b in windows(s, e, self.win, self.hop):
            x = self._slice(a, b)
            if x.size < 0.5 * 16000:
                continue
            level = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
            if level < self.silence_dbfs:
                out.append(self.result(a, b, [], M.insufficient_data, quality=Quality(notes=["тишина"])))
                continue
            r = await self.run_blocking(self.backend.infer, x, gated=True)
            out.append(self._to_result(a, b, r, event.segment_id))
        return out

    def _to_result(self, a: float, b: float, r: dict[str, Any], seg_id: int) -> AnalysisResult:
        labels = [str(lab).split("/")[-1].strip("<>").lower() for lab in r["labels"]]  # «生气/angry» → «angry»
        scores = dict(zip(labels, r["scores"], strict=False))
        scores.pop("unk", None)
        top = max(scores, key=scores.get) if scores else None
        meth = f"{self.model_id}: softmax классификационной головы (не калиброван)"
        metrics = [Metric(name="top_label", value=top, method=meth, kind="model_output", group="predicted_voice_expression")]
        for lab, sc in scores.items():
            metrics.append(Metric(name=f"score_{lab}", value=round(sc, 5), unit="model score", method=meth, kind="model_output", group="predicted_voice_expression"))
        if "happy" in scores:
            pmn = scores["happy"] - sum(scores.get(k, 0.0) for k in NEGATIVE)
            metrics.append(Metric(name="positive_minus_negative", value=round(pmn, 5), unit="score diff", method="happy − (angry+sad+fearful+disgusted); исследовательская производная", kind="experimental", group="predicted_voice_expression"))
        payload: dict[str, Any] = {"segment_id": seg_id, "validation": "not_validated_ru", "scores": scores}
        if self.save_emb and r.get("embedding") is not None:
            payload["raw"] = [{"t_start_ms": int(a), "t_end_ms": int(b), "embedding": [round(float(v), 6) for v in r["embedding"]]}]
            payload["embedding_dim"] = int(r["embedding"].size)
        return self.result(a, b, metrics, M.ok, quality=Quality(notes=[VALIDATION_NOTE]), payload=payload)
