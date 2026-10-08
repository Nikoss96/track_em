"""Опциональная запись исходной медиаинформации Live-сессии (только при save_media=true)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import av
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class MediaRecorder:
    def __init__(self, out_dir: Path, video_fps: int = 10) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.video_fps = video_fps
        self._lock = threading.Lock()
        self._wav: sf.SoundFile | None = None
        self._container: av.container.OutputContainer | None = None
        self._stream = None
        self._last_v_ms = -1e9
        self.closed = False

    def write_audio(self, samples: np.ndarray, sr: int, t_ms: float) -> None:
        with self._lock:
            if self.closed:
                return
            if self._wav is None:
                self._wav = sf.SoundFile(str(self.out_dir / "audio.wav"), "w", samplerate=sr, channels=samples.shape[1], subtype="PCM_16")
            self._wav.write(samples)

    def write_video(self, frame: np.ndarray, t_ms: float) -> None:
        if t_ms - self._last_v_ms < 1000 / self.video_fps:
            return
        with self._lock:
            if self.closed:
                return
            self._last_v_ms = t_ms
            try:
                if self._container is None:
                    h, w = frame.shape[:2]
                    self._container = av.open(str(self.out_dir / "video.mp4"), "w")
                    self._stream = self._container.add_stream("mpeg4", rate=self.video_fps)
                    self._stream.width, self._stream.height = w - w % 2, h - h % 2
                    self._stream.pix_fmt = "yuv420p"
                    self._stream.bit_rate = 2_000_000
                f = av.VideoFrame.from_ndarray(frame[: self._stream.height, : self._stream.width], format="rgb24")
                for p in self._stream.encode(f):
                    self._container.mux(p)
            except Exception as e:  # noqa: BLE001
                log.warning("media recorder video: %s", e)

    def close(self) -> list[str]:
        with self._lock:
            self.closed = True
            files = []
            if self._wav is not None:
                self._wav.close()
                files.append(str(self.out_dir / "audio.wav"))
            if self._container is not None and self._stream is not None:
                try:
                    for p in self._stream.encode():
                        self._container.mux(p)
                    self._container.close()
                    files.append(str(self.out_dir / "video.mp4"))
                except Exception as e:  # noqa: BLE001
                    log.warning("media recorder close: %s", e)
            return files
