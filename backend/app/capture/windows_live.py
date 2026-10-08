"""WindowsLiveSource: захват области экрана (DXcam) и системного звука (WASAPI loopback, SoundCard).

Аудио-время считается по числу полученных сэмплов (непрерывная ось),
привязанной к монотонным часам при первом чанке; расхождение с часами
(drift) измеряется и отдаётся в статусе.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Any

import numpy as np

from app.capture.base import MediaSource, SourceSink

log = logging.getLogger(__name__)


def _com_init() -> None:
    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


class WindowsLiveSource(MediaSource):
    kind = "live"

    def __init__(
        self,
        monitor: int = 0,
        region: tuple[int, int, int, int] | None = None,  # left, top, right, bottom
        audio_device: str | None = None,
        fps: int = 30,
        audio_chunk_ms: int = 200,
        audio_rate: int = 48000,
        capture_video: bool = True,
        capture_audio: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.region = region
        self.audio_device = audio_device
        self.fps = fps
        self.audio_chunk_ms = audio_chunk_ms
        self.audio_rate = audio_rate
        self.capture_video = capture_video
        self.capture_audio = capture_audio
        self._camera: Any = None
        self.drift_ms = 0.0
        self.video_frames = 0
        self.audio_chunks = 0

    def _run_threads(self, sink: SourceSink, clock: Any) -> list[threading.Thread]:
        th = []
        if self.capture_video:
            th.append(threading.Thread(target=self._video_loop, args=(sink, clock), name="live-video", daemon=True))
        if self.capture_audio:
            th.append(threading.Thread(target=self._audio_loop, args=(sink, clock), name="live-audio", daemon=True))
        return th

    # --- видео ------------------------------------------------------------
    def _video_loop(self, sink: SourceSink, clock: Any) -> None:
        try:
            import dxcam

            from app.capture import dxcam_patch

            dxcam_patch.apply()
            self._camera = dxcam.create(output_idx=self.monitor, output_color="RGB")
            if self._camera is None:
                raise RuntimeError("dxcam.create вернул None (монитор недоступен?)")
        except Exception as e:  # noqa: BLE001
            sink.on_error("live.video", f"init: {e}", fatal=False)
            return
        period = 1.0 / self.fps
        last: np.ndarray | None = None
        next_t = time.perf_counter()
        fails = 0
        try:
            while not self._stop.is_set():
                t_ns = time.monotonic_ns()
                try:
                    frame = self._camera.grab(region=self.region)
                    fails = 0
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails in (1, 50):
                        sink.on_error("live.video", f"grab: {e}")
                    frame = None
                    time.sleep(0.05)
                # DXcam возвращает None, если экран не менялся: повторяем последний кадр.
                if frame is None:
                    frame = last
                if frame is not None:
                    last = frame
                    self.video_frames += 1
                    sink.on_video(frame, clock.from_monotonic_ns(t_ns))
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        finally:
            self._release_camera()

    def _release_camera(self) -> None:
        cam, self._camera = self._camera, None
        if cam is not None:
            try:
                cam.release()
            except Exception:  # noqa: BLE001
                pass
            try:
                del cam
            except Exception:  # noqa: BLE001
                pass

    # --- аудио ------------------------------------------------------------
    def _audio_loop(self, sink: SourceSink, clock: Any) -> None:
        _com_init()
        try:
            import soundcard as sc

            spk = find_speaker(self.audio_device)
            mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
            n = int(self.audio_rate * self.audio_chunk_ms / 1000)
            total = 0
            t0_ms: float | None = None
            with mic.recorder(samplerate=self.audio_rate, blocksize=n // 2) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=n)
                    arrival_ms = clock.now_ms()
                    if data is None or len(data) == 0:
                        continue
                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim == 1:
                        data = data[:, None]
                    dur_ms = data.shape[0] / self.audio_rate * 1000
                    if t0_ms is None:
                        t0_ms = max(0.0, arrival_ms - dur_ms)
                    t_ms = t0_ms + total / self.audio_rate * 1000
                    total += data.shape[0]
                    self.drift_ms = arrival_ms - (t_ms + dur_ms)
                    self.audio_chunks += 1
                    sink.on_audio(data, self.audio_rate, t_ms)
        except Exception as e:  # noqa: BLE001
            log.exception("loopback failed")
            sink.on_error("live.audio", f"{type(e).__name__}: {e}", fatal=False)

    def release(self) -> None:
        self._release_camera()


def find_speaker(name: str | None) -> Any:
    import soundcard as sc

    if name:
        for s in sc.all_speakers():
            if s.name == name or str(s.id) == name:
                return s
        raise RuntimeError(f"аудиоустройство не найдено: {name}")
    spk = sc.default_speaker()
    if spk is None:
        raise RuntimeError("нет выходного аудиоустройства по умолчанию")
    return spk
