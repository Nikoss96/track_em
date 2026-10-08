"""Windows smoke-test Live Capture: DXcam + WASAPI loopback, N секунд.

    .venv\\Scripts\\python.exe scripts\\smoke_live.py --seconds 5 [--monitor 0] [--region 0,0,640,480] [--device "Speakers ..."]

Выводит число кадров/аудио-чанков, фактический FPS, drift аудио-часов, ошибки,
и проверяет, что устройства освобождены (повторный старт работает).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from app.capture.devices import list_audio_outputs, list_monitors  # noqa: E402
from app.capture.windows_live import WindowsLiveSource  # noqa: E402
from app.streaming.clock import SessionClock  # noqa: E402


class Sink:
    def __init__(self) -> None:
        self.frames = 0
        self.shape = None
        self.audio_s = 0.0
        self.audio_rate = None
        self.peak = 0.0
        self.errors: list[str] = []
        self.audio_t: list[float] = []
        self.lock = threading.Lock()

    def on_video(self, frame, t_ms, pts_ms=None):
        self.frames += 1
        self.shape = frame.shape

    def on_audio(self, samples, sample_rate, t_ms, pts_ms=None):
        with self.lock:
            self.audio_s += samples.shape[0] / sample_rate
            self.audio_rate = sample_rate
            self.peak = max(self.peak, float(np.abs(samples).max()))
            self.audio_t.append(t_ms)

    def on_error(self, where, message, fatal=False):
        self.errors.append(f"{where}: {message}")

    def on_eof(self):
        pass


def run_once(args, seconds: float) -> dict:
    region = tuple(int(v) for v in args.region.split(",")) if args.region else None
    if region:
        region = (region[0], region[1], region[0] + region[2], region[1] + region[3])
    src = WindowsLiveSource(monitor=args.monitor, region=region, audio_device=args.device, fps=30)
    clock = SessionClock()
    clock.start()
    sink = Sink()
    src.start(sink, clock)
    time.sleep(seconds)
    src.stop()
    return {
        "frames": sink.frames,
        "fps": round(sink.frames / seconds, 1),
        "frame_shape": sink.shape,
        "audio_seconds": round(sink.audio_s, 2),
        "audio_rate": sink.audio_rate,
        "audio_peak": round(sink.peak, 4),
        "drift_ms": round(src.drift_ms, 1),
        "threads_alive": src.alive,
        "errors": sink.errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--monitor", type=int, default=0)
    ap.add_argument("--region", default=None, help="left,top,width,height")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    print("monitors:", list_monitors())
    print("audio outputs:", list_audio_outputs())
    r1 = run_once(args, args.seconds)
    print("run 1:", r1)
    r2 = run_once(args, 2)  # повторный старт: устройства должны быть освобождены
    print("run 2:", r2)
    ok = r1["frames"] > 0 and r1["audio_seconds"] > 0.5 * args.seconds and not r1["threads_alive"] and r2["frames"] > 0
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
