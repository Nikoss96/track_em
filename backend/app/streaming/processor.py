"""StreamProcessor: приёмник сырых данных источника → унифицированные события.

- видео → RGB uint8, контроль пустых/повреждённых кадров, превью JPEG;
- аудио → mono float32 16 kHz (soxr), чанки фиксированной длины, RMS dBFS;
- общая ось времени (мс от session_t0), обнаружение разрывов аудио;
- публикация в EventRouter из потоков источника через call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import soxr

from app.schemas.events import AudioChunk, EventType, ModuleErrorEvent, VideoFrame
from app.streaming.clock import SessionClock
from app.streaming.router import EventRouter

log = logging.getLogger(__name__)
TARGET_SR = 16000


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(max(r, 1e-6))


class StreamProcessor:
    def __init__(
        self,
        session_id: str,
        router: EventRouter,
        loop: asyncio.AbstractEventLoop,
        clock: SessionClock,
        cfg: dict[str, Any],
        on_eof: Callable[[], None] | None = None,
        media_recorder: Any = None,
        backpressure: bool = False,
    ) -> None:
        # backpressure=True (File Replay не в реальном времени): источник ждёт, пока очереди
        # анализаторов не освободятся — без потерь кадров, результаты воспроизводимы.
        self.backpressure = backpressure
        self.backpressure_wait_ms = 0.0
        cap = cfg["capture"]
        self.session_id = session_id
        self.router = router
        self.loop = loop
        self.clock = clock
        self.chunk_samples = int(TARGET_SR * cap["audio_chunk_ms"] / 1000)
        self.silence_dbfs = float(cap["silence_dbfs"])
        self.blank_std = float(cap["blank_frame_std"])
        self.preview_period_ms = 1000.0 / max(cap["preview_fps"], 0.1)
        self.preview_width = int(cap["preview_width"])
        self._on_eof = on_eof
        self.media_recorder = media_recorder
        self._lock = threading.Lock()
        # видео
        self.video_seq = 0
        self.video_corrupted = 0
        self.blank_frames = 0
        self.last_video_t_ms: float | None = None
        self.last_video_wall = 0.0
        self._fps_times: deque[float] = deque(maxlen=60)
        self.latest_preview: bytes | None = None
        self.latest_preview_t_ms: float | None = None
        self._last_preview_t = -1e9
        self.frame_size: tuple[int, int] | None = None
        # аудио
        self._resampler: soxr.ResampleStream | None = None
        self._src_rate: int | None = None
        self._src_channels: int | None = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_t_ms: float | None = None  # время первого сэмпла в _buf
        self._expected_in_ms: float | None = None
        self._pending_gap_ms = 0.0
        self.audio_seq = 0
        self.audio_gaps = 0
        self.audio_gap_ms = 0.0
        self.silent_ms = 0.0
        self.levels: deque[tuple[int, float]] = deque(maxlen=50)
        self.last_audio_wall = 0.0
        self.audio_samples_out = 0
        # общее
        self.errors: deque[dict[str, Any]] = deque(maxlen=50)
        self.eof = False
        self.capture_lag_ms = 0.0

    # --- публикация --------------------------------------------------------
    def _publish(self, ev: Any) -> None:
        try:
            self.loop.call_soon_threadsafe(self.router.publish, ev)
        except RuntimeError:  # loop закрыт
            pass

    def _wait_capacity(self, etype: EventType, timeout_s: float = 120.0) -> None:
        if not self.backpressure:
            return
        t0 = time.monotonic()
        while self.router.saturated(etype) and time.monotonic() - t0 < timeout_s:
            time.sleep(0.003)
        self.backpressure_wait_ms += (time.monotonic() - t0) * 1000

    # --- SourceSink --------------------------------------------------------
    def on_video(self, frame: np.ndarray, t_ms: float, pts_ms: float | None = None) -> None:
        self._wait_capacity(EventType.video_frame)
        corrupted = False
        if frame is None or frame.ndim != 3 or frame.shape[2] not in (3, 4) or frame.size == 0:
            self.video_corrupted += 1
            return
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        # «нет изображения»: почти однотонный кадр (оценка по уменьшенной копии)
        small = frame[::16, ::16]
        if float(small.std()) < self.blank_std:
            self.blank_frames += 1
            corrupted = True
        now = time.monotonic()
        self._fps_times.append(now)
        self.last_video_wall = now
        self.last_video_t_ms = t_ms
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.capture_lag_ms = self.clock.now_ms() - t_ms
        self.video_seq += 1
        ev = VideoFrame(
            type=EventType.video_frame,
            session_id=self.session_id,
            t_ms=int(round(t_ms)),
            frame=frame,
            seq=self.video_seq,
            source_pts_ms=pts_ms,
            corrupted=corrupted,
        )
        self._publish(ev)
        if self.media_recorder is not None:
            self.media_recorder.write_video(frame, t_ms)
        if t_ms - self._last_preview_t >= self.preview_period_ms:
            self._last_preview_t = t_ms
            self._make_preview(frame, t_ms)

    def _make_preview(self, frame: np.ndarray, t_ms: float) -> None:
        h, w = frame.shape[:2]
        scale = self.preview_width / w if w > self.preview_width else 1.0
        img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.latest_preview = buf.tobytes()
            self.latest_preview_t_ms = t_ms

    def on_audio(self, samples: np.ndarray, sample_rate: int, t_ms: float, pts_ms: float | None = None) -> None:
        self._wait_capacity(EventType.audio_chunk)
        with self._lock:
            self._on_audio(samples, sample_rate, t_ms, pts_ms)

    def _on_audio(self, samples: np.ndarray, sr: int, t_ms: float, pts_ms: float | None) -> None:
        if samples.ndim == 1:
            samples = samples[:, None]
        if self.media_recorder is not None:
            self.media_recorder.write_audio(samples, sr, t_ms)
        n_in = samples.shape[0]
        dur_ms = n_in / sr * 1000.0
        self.last_audio_wall = time.monotonic()
        mono = samples.mean(axis=1).astype(np.float32) if samples.shape[1] > 1 else samples[:, 0].astype(np.float32)
        if self._resampler is None or sr != self._src_rate:
            self._src_rate = sr
            self._src_channels = samples.shape[1]
            self._resampler = None if sr == TARGET_SR else soxr.ResampleStream(sr, TARGET_SR, 1, dtype="float32", quality="HQ")
        # разрыв во входном потоке (потеря сэмплов на стороне источника)
        if self._expected_in_ms is not None:
            gap = t_ms - self._expected_in_ms
            if gap > 40.0:
                self.audio_gaps += 1
                self.audio_gap_ms += gap
                self._flush(final=False)
                self._pending_gap_ms += gap
                self._buf_t_ms = None
        self._expected_in_ms = t_ms + dur_ms
        out = self._resampler.resample_chunk(mono) if self._resampler is not None else mono
        if self._buf_t_ms is None:
            self._buf_t_ms = t_ms
        self._buf = np.concatenate([self._buf, out]) if self._buf.size else out
        while self._buf.size >= self.chunk_samples:
            chunk, self._buf = self._buf[: self.chunk_samples], self._buf[self.chunk_samples :]
            self._emit_chunk(chunk)

    def _emit_chunk(self, chunk: np.ndarray) -> None:
        assert self._buf_t_ms is not None
        dur = chunk.size / TARGET_SR * 1000.0
        level = rms_dbfs(chunk)
        self.audio_seq += 1
        ev = AudioChunk(
            type=EventType.audio_chunk,
            session_id=self.session_id,
            t_ms=int(round(self._buf_t_ms)),
            pcm16k=np.ascontiguousarray(chunk, dtype=np.float32),
            duration_ms=dur,
            seq=self.audio_seq,
            source_rate=self._src_rate or TARGET_SR,
            source_channels=self._src_channels or 1,
            rms_dbfs=level,
            gap_before_ms=self._pending_gap_ms,
        )
        self._pending_gap_ms = 0.0
        self._buf_t_ms += dur
        self.audio_samples_out += chunk.size
        self.levels.append((ev.t_ms, round(level, 1)))
        self.silent_ms = self.silent_ms + dur if level < self.silence_dbfs else 0.0
        self._publish(ev)

    def _flush(self, final: bool) -> None:
        if self._resampler is not None and final:
            tail = self._resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            self._resampler = None  # поток ресемплера завершён; новый создаётся при новых данных
            if tail.size:
                self._buf = np.concatenate([self._buf, tail])
        if self._buf.size >= TARGET_SR // 20 and self._buf_t_ms is not None:  # ≥50 мс
            self._emit_chunk(self._buf)
        self._buf = np.zeros(0, dtype=np.float32)

    def on_error(self, where: str, message: str, fatal: bool = False) -> None:
        self.errors.append({"where": where, "message": message, "fatal": fatal, "t_ms": round(self.clock.now_ms())})
        log.warning("capture error %s: %s", where, message)
        self._publish(
            ModuleErrorEvent(
                type=EventType.module_error,
                session_id=self.session_id,
                t_ms=int(self.clock.now_ms()),
                module_id=f"capture:{where}",
                message=message,
                fatal=fatal,
            )
        )

    def on_eof(self) -> None:
        with self._lock:
            self._flush(final=True)
        self.eof = True
        if self._on_eof:
            try:
                self.loop.call_soon_threadsafe(self._on_eof)
            except RuntimeError:
                pass

    def finish(self) -> None:
        with self._lock:
            self._flush(final=True)

    # --- статус -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        fps = 0.0
        if len(self._fps_times) > 1:
            span = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0
        video_stale = self.last_video_wall == 0 or now - self.last_video_wall > 2.0
        audio_stale = self.last_audio_wall == 0 or now - self.last_audio_wall > 2.0
        return {
            "t_ms": round(self.clock.now_ms()),
            "video": {
                "frames": self.video_seq,
                "fps": round(fps, 1),
                "size": self.frame_size,
                "blank_frames": self.blank_frames,
                "corrupted": self.video_corrupted,
                "no_signal": video_stale,
                "capture_lag_ms": round(self.capture_lag_ms, 1),
            },
            "audio": {
                "chunks": self.audio_seq,
                "source_rate": self._src_rate,
                "source_channels": self._src_channels,
                "seconds": round(self.audio_samples_out / TARGET_SR, 2),
                "gaps": self.audio_gaps,
                "gap_ms": round(self.audio_gap_ms, 1),
                "silence": self.silent_ms >= 2000,
                "silent_ms": round(self.silent_ms),
                "no_signal": audio_stale,
                "levels": list(self.levels)[-20:],
            },
            "eof": self.eof,
            "backpressure": {"enabled": self.backpressure, "wait_ms": round(self.backpressure_wait_ms)},
            "errors": list(self.errors)[-5:],
        }
