"""FileReplaySource: MP4/MKV/WAV → потоковые события через PyAV (FFmpeg).

Временная ось — media PTS (мс) относительно минимального стартового PTS потоков.
В режиме realtime доставка синхронизируется с часами сессии (с учётом speed).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import av
import numpy as np

from app.capture.base import MediaSource, SourceSink

log = logging.getLogger(__name__)


class FileReplaySource(MediaSource):
    kind = "file"

    def __init__(
        self,
        path: str,
        realtime: bool = True,
        speed: float = 1.0,
        video_fps: float = 30.0,
        max_width: int = 1280,
        max_duration_s: float | None = None,
    ) -> None:
        super().__init__()
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.realtime = realtime
        self.speed = speed
        self.video_fps = video_fps
        self.max_width = max_width
        self.max_duration_s = max_duration_s
        self.info = probe(self.path)

    def _run_threads(self, sink: SourceSink, clock: Any) -> list[threading.Thread]:
        return [threading.Thread(target=self._loop, args=(sink, clock), name="file-replay", daemon=True)]

    def _pace(self, clock: Any, t_ms: float) -> None:
        if not self.realtime:
            return
        target = t_ms / self.speed
        while not self._stop.is_set():
            lag = target - clock.now_ms()
            if lag <= 0:
                return
            time.sleep(min(lag, 50) / 1000)

    def _loop(self, sink: SourceSink, clock: Any) -> None:
        try:
            container = av.open(self.path)
        except Exception as e:  # noqa: BLE001
            sink.on_error("file", f"open failed: {e}", fatal=True)
            sink.on_eof()
            return
        try:
            streams = []
            vstream = container.streams.video[0] if container.streams.video else None
            astream = container.streams.audio[0] if container.streams.audio else None
            if vstream is not None:
                vstream.thread_type = "AUTO"
                streams.append(vstream)
            if astream is not None:
                streams.append(astream)
            if not streams:
                sink.on_error("file", "no audio/video streams", fatal=True)
                return
            # Базовый PTS — минимальный start_time среди потоков.
            starts = [
                float(s.start_time * s.time_base) for s in streams if s.start_time is not None
            ]
            base_s = min(starts) if starts else 0.0
            resampler = (
                av.AudioResampler(format="flt", layout=astream.layout.name, rate=astream.rate)
                if astream is not None
                else None
            )
            min_dt_ms = 1000.0 / self.video_fps
            last_v_ms = -1e9
            for frame in container.decode(*streams):
                if self._stop.is_set():
                    break
                if frame.pts is None:
                    continue
                t_ms = (float(frame.pts * frame.time_base) - base_s) * 1000.0
                if self.max_duration_s is not None and t_ms > self.max_duration_s * 1000:
                    break
                if isinstance(frame, av.VideoFrame):
                    if t_ms - last_v_ms < min_dt_ms - 1:
                        continue  # прореживание до целевого FPS
                    last_v_ms = t_ms
                    if frame.width > self.max_width:
                        h = int(frame.height * self.max_width / frame.width) // 2 * 2
                        frame = frame.reformat(width=self.max_width, height=h)
                    try:
                        rgb = frame.to_ndarray(format="rgb24")
                    except Exception as e:  # noqa: BLE001
                        sink.on_error("file.video", f"decode: {e}")
                        continue
                    self._pace(clock, t_ms)
                    sink.on_video(rgb, t_ms, pts_ms=t_ms)
                elif isinstance(frame, av.AudioFrame) and resampler is not None:
                    for rf in resampler.resample(frame):
                        arr = rf.to_ndarray()  # (1, n*ch) packed float32
                        ch = len(rf.layout.channels)
                        samples = arr.reshape(-1, ch).astype(np.float32, copy=False)
                        self._pace(clock, t_ms)
                        sink.on_audio(samples, rf.sample_rate, t_ms, pts_ms=t_ms)
                        t_ms += samples.shape[0] / rf.sample_rate * 1000.0
        except Exception as e:  # noqa: BLE001
            log.exception("file replay failed")
            sink.on_error("file", f"{type(e).__name__}: {e}", fatal=True)
        finally:
            container.close()
            sink.on_eof()


def probe(path: str) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    with av.open(str(path)) as c:
        info["duration_s"] = float(c.duration / av.time_base) if c.duration else None
        if c.streams.video:
            v = c.streams.video[0]
            info["video"] = {
                "width": v.codec_context.width,
                "height": v.codec_context.height,
                "fps": float(v.average_rate) if v.average_rate else None,
                "codec": v.codec_context.name,
            }
        if c.streams.audio:
            a = c.streams.audio[0]
            info["audio"] = {
                "rate": a.rate,
                "channels": a.channels if hasattr(a, "channels") else len(a.layout.channels),
                "codec": a.codec_context.name,
            }
    return info
