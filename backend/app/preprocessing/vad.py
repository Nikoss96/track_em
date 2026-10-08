"""Потоковый VAD на Silero VAD v6 (ONNX из поставки faster-whisper) и сегментатор речи."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.events import AudioChunk, EventType, SpeechActivity
from app.streaming.router import EventRouter, Subscription

log = logging.getLogger(__name__)
FRAME = 512  # 32 мс при 16 кГц
CTX = 64


class StreamingSileroVAD:
    """Stateful Silero: сохраняет h/c и контекст между вызовами."""

    model_id = "silero_vad_v6"

    def __init__(self) -> None:
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(CTX, dtype=np.float32)
        self.rest = np.zeros(0, dtype=np.float32)

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        """Возвращает вероятности речи для каждого полного 512-сэмпл кадра."""
        x = np.concatenate([self.rest, pcm.astype(np.float32)])
        n = x.size // FRAME
        self.rest = x[n * FRAME :]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = x[: n * FRAME].reshape(n, FRAME)
        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            inp = np.concatenate([self.context, frames[i]])[None, :]
            out, self.h, self.c = self.session.run(None, {"input": inp, "h": self.h, "c": self.c})
            probs[i] = float(np.asarray(out).reshape(-1)[0])
            self.context = frames[i][-CTX:]
        return probs


@dataclass
class _Seg:
    start_ms: float
    probs: list[float]
    last_speech_ms: float


class VadStage:
    """Подписка на audio.chunk → публикация speech.activity (start/end сегментов)."""

    def __init__(self, session_id: str, router: EventRouter, cfg: dict[str, Any]) -> None:
        sp = cfg["modules"]["speech"]
        self.session_id = session_id
        self.router = router
        self.threshold = float(sp["vad_threshold"])
        self.neg_threshold = max(self.threshold - 0.15, 0.01)
        self.min_silence_ms = float(sp["min_silence_ms"])
        self.min_speech_ms = float(sp["min_speech_ms"])
        self.max_segment_ms = float(sp["max_segment_ms"])
        self.vad: StreamingSileroVAD | None = None
        self.sub: Subscription | None = None
        self.task: asyncio.Task[None] | None = None
        self.seg: _Seg | None = None
        self.segment_id = 0
        self.segments = 0
        self.speech_ms = 0.0
        self.error: str | None = None
        self._next_t: float | None = None

    def start(self) -> None:
        try:
            self.vad = StreamingSileroVAD()
        except Exception as e:  # noqa: BLE001
            self.error = f"VAD недоступен: {e}"
            log.error(self.error)
            return
        self.sub = self.router.subscribe("vad", [EventType.audio_chunk], maxsize=2000, policy="lossless")
        self.task = asyncio.create_task(self._run(), name="vad")

    async def _run(self) -> None:
        assert self.sub is not None
        async for ev in self.sub:
            if isinstance(ev, AudioChunk):
                self.feed(ev)
        self.flush()

    def feed(self, ch: AudioChunk) -> None:
        assert self.vad is not None
        if ch.gap_before_ms > 0 or self._next_t is None:
            if ch.gap_before_ms > 0 and self.seg is not None:
                self._end(self._next_t or ch.t_ms, forced=False)
            self.vad.reset()
            self._next_t = float(ch.t_ms)
        frame_ms = FRAME / 16.0
        t0 = self._next_t - self.vad.rest.size / 16.0
        probs = self.vad(ch.pcm16k)
        self._next_t = float(ch.t_ms) + ch.duration_ms
        for i, p in enumerate(probs):
            t = t0 + i * frame_ms
            self._step(float(p), t, t + frame_ms)

    def _step(self, p: float, t: float, t_end: float) -> None:
        if self.seg is None:
            if p >= self.threshold:
                self.seg = _Seg(start_ms=t, probs=[p], last_speech_ms=t_end)
                self.segment_id += 1
                self._pub("start", t, t, forced=False)
            return
        self.seg.probs.append(p)
        if p >= self.neg_threshold:
            self.seg.last_speech_ms = t_end
        if t_end - self.seg.last_speech_ms >= self.min_silence_ms:
            self._end(self.seg.last_speech_ms, forced=False)
        elif t_end - self.seg.start_ms >= self.max_segment_ms:
            self._end(t_end, forced=True)
            self.seg = _Seg(start_ms=t_end, probs=[], last_speech_ms=t_end)
            self.segment_id += 1
            self._pub("start", t_end, t_end, forced=True)

    def _end(self, t_end: float, forced: bool) -> None:
        seg, self.seg = self.seg, None
        if seg is None:
            return
        dur = t_end - seg.start_ms
        if dur < self.min_speech_ms and not forced:
            # короткий всплеск — не речь; сообщаем об отмене сегмента
            self._pub("cancel", seg.start_ms, t_end, forced=False)
            return
        self.segments += 1
        self.speech_ms += dur
        mean_p = float(np.mean(seg.probs)) if seg.probs else None
        self._pub("end", seg.start_ms, t_end, forced=forced, mean_p=mean_p)

    def _pub(self, kind: str, t: float, t_end: float, forced: bool, mean_p: float | None = None) -> None:
        self.router.publish(
            SpeechActivity(
                type=EventType.speech_activity,
                session_id=self.session_id,
                t_ms=int(round(t)),
                kind=kind,
                t_end_ms=int(round(t_end)),
                segment_id=self.segment_id,
                forced=forced,
                mean_probability=mean_p,
            )
        )

    def flush(self) -> None:
        if self.seg is not None:
            self._end(self.seg.last_speech_ms, forced=False)

    async def stop(self) -> None:
        if self.sub is not None and not self.sub.closed:
            self.sub.close()
        if self.task is not None:
            await self.task

    def status(self) -> dict[str, Any]:
        return {
            "model": StreamingSileroVAD.model_id,
            "in_speech": self.seg is not None,
            "segments": self.segments,
            "speech_ms": round(self.speech_ms),
            "error": self.error,
            "queue": self.sub.qsize() if self.sub else 0,
        }
