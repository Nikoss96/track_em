"""Module A: FaceAnalyzer — MediaPipe Face Landmarker blendshapes.

Выход по окнам window_ms: средние ключевых групп blendshapes, интенсивность и изменчивость
экспрессии, качество детекции. Сырые коэффициенты по кадрам сохраняются (payload.raw → Parquet).
Blendshapes ≠ FACS Action Units; интерпретация как «эмоций» не производится.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analyzers.base import AnalyzerPlugin, ModelUnavailable
from app.analyzers.face.backends import FaceFrame, MediaPipeFaceBackend, PyFeatBackend
from app.config import models_dir
from app.schemas.events import Event, EventType, VideoFrame
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus
from app.storage.models import ensure_file, sha256_file

# Производные группы (среднее соответствующих коэффициентов MediaPipe)
GROUPS: dict[str, tuple[str, ...]] = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "brow_raise": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "brow_down": ("browDownLeft", "browDownRight"),
    "mouth_open": ("jawOpen",),
    "lip_press": ("mouthPressLeft", "mouthPressRight"),
    "lip_stretch": ("mouthStretchLeft", "mouthStretchRight"),
    "lip_pucker": ("mouthPucker",),
    "lip_funnel": ("mouthFunnel",),
    "lip_corner_down": ("mouthFrownLeft", "mouthFrownRight"),
    "eye_blink": ("eyeBlinkLeft", "eyeBlinkRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),
    "cheek_squint": ("cheekSquintLeft", "cheekSquintRight"),
    "nose_sneer": ("noseSneerLeft", "noseSneerRight"),
}
# Для интенсивности экспрессии исключаем нейтральный класс, направление взгляда и моргание.
EXCLUDE_INTENSITY = ("_neutral", "eyeLook", "eyeBlink")


def expression_vector(bs: dict[str, float]) -> tuple[list[str], np.ndarray]:
    keys = sorted(k for k in bs if not k.startswith(EXCLUDE_INTENSITY))
    return keys, np.array([bs[k] for k in keys], dtype=np.float32)


class FaceAnalyzer(AnalyzerPlugin):
    module_id = "face"
    version = "0.1.0"
    subscribes = (EventType.video_frame,)
    queue_policy = "drop_oldest"
    queue_size = 2

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.period_ms = 1000.0 / float(config.get("analysis_fps", 10))
        self.window_ms = int(config.get("window_ms", 1000))
        backend = config.get("backend", "mediapipe")
        if backend == "pyfeat":
            PyFeatBackend()
        url = config["model_url"]
        path = models_dir(self.ctx.config) / "mediapipe" / Path(url).name
        try:
            await self.run_blocking(ensure_file, url, path, gated=False)
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"не удалось получить {url}: {e}") from e
        self.backend = await self.run_blocking(MediaPipeFaceBackend, path, float(config.get("min_face_presence", 0.5)), gated=False)
        self.model_id = "mediapipe/face_landmarker.task (float16 v1)"
        self.model_version = f"mediapipe {self.backend.version}"
        sha = await self.run_blocking(sha256_file, path, gated=False)
        self.ctx.shared["register_model"](
            model_id="mediapipe/face_landmarker", version="float16/1", backend=f"mediapipe {self.backend.version}",
            source=url, license="Apache-2.0", languages="n/a", local_path=str(path), sha256=sha,
            notes="52 blendshapes (ARKit-подобные), не FACS AU",
        )
        self._last_t = -1e9
        self._win_start: int | None = None
        self._frames: list[dict[str, Any]] = []
        self._prev_vec: np.ndarray | None = None
        self._lost_frames = 0

    async def process(self, event: Event) -> list[AnalysisResult]:
        assert isinstance(event, VideoFrame)
        if event.t_ms - self._last_t < self.period_ms - 1:
            return []
        self._last_t = event.t_ms
        out: list[AnalysisResult] = []
        if self._win_start is None:
            self._win_start = event.t_ms - event.t_ms % self.window_ms
        while event.t_ms >= self._win_start + self.window_ms:
            out.append(self._emit_window())
        rec = await self.run_blocking(self._analyze, event.frame, event.t_ms, event.corrupted, gated=False)
        self._frames.append(rec)
        return out

    def _analyze(self, rgb: np.ndarray, t_ms: int, corrupted: bool) -> dict[str, Any]:
        ff: FaceFrame = self.backend(rgb, t_ms)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rec: dict[str, Any] = {
            "t_ms": t_ms,
            "face": ff.face,
            "brightness": float(gray.mean()),
            "sharpness": float(cv2.Laplacian(gray[::2, ::2], cv2.CV_32F).var()),
            "corrupted_input": corrupted,
        }
        if ff.face:
            rec.update({"yaw": ff.yaw, "pitch": ff.pitch, "roll": ff.roll, "face_size": ff.face_size})
            rec["blendshapes"] = {k: round(v, 5) for k, v in ff.blendshapes.items()}
            _, vec = expression_vector(ff.blendshapes)
            rec["expression_intensity"] = float(vec.mean()) if vec.size else None
            for g, keys in GROUPS.items():
                vals = [ff.blendshapes[k] for k in keys if k in ff.blendshapes]
                rec[g] = float(np.mean(vals)) if vals else None
            rec["motion"] = float(np.abs(vec - self._prev_vec).mean()) if self._prev_vec is not None and self._prev_vec.size == vec.size else None
            self._prev_vec = vec
        else:
            self._prev_vec = None
        return rec

    def _emit_window(self) -> AnalysisResult:
        assert self._win_start is not None
        ws, we = self._win_start, self._win_start + self.window_ms
        frames, self._frames = self._frames, []
        self._win_start = we
        n = len(frames)
        with_face = [f for f in frames if f["face"]]
        quality = Quality(input_ok=n > 0)
        metrics: list[Metric] = [
            Metric(name="frames_analyzed", value=n, unit="frames", method="count", kind="measured", group="quality"),
            Metric(name="face_presence", value=round(len(with_face) / n, 4) if n else 0.0, unit="ratio", method="доля кадров с обнаруженным лицом", kind="measured", group="quality"),
        ]
        if n:
            metrics.append(Metric(name="brightness", value=round(float(np.mean([f["brightness"] for f in frames])), 2), unit="0-255", method="средняя яркость", group="quality"))
            metrics.append(Metric(name="sharpness", value=round(float(np.mean([f["sharpness"] for f in frames])), 2), unit="laplacian var", method="дисперсия лапласиана", group="quality"))
        if not with_face:
            self._lost_frames += n
            if n:
                quality.notes.append("лицо не обнаружено")
            status = ResultStatus.insufficient_data
            return self.result(ws, we, metrics, status, quality=quality, payload={"raw": frames})
        self._lost_frames = 0
        method_bs = "MediaPipe blendshape coefficient, среднее по кадрам окна"
        for g in GROUPS:
            vals = [f[g] for f in with_face if f.get(g) is not None]
            if vals:
                metrics.append(Metric(name=g, value=round(float(np.mean(vals)), 5), unit="coef 0-1", method=method_bs, kind="model_output", group="blendshape_groups"))
        inten = [f["expression_intensity"] for f in with_face if f.get("expression_intensity") is not None]
        if inten:
            metrics.append(Metric(name="expression_intensity", value=round(float(np.mean(inten)), 5), unit="coef 0-1", method="среднее экспрессивных blendshapes (без neutral/взгляда/моргания)", kind="derived", group="expressivity"))
            metrics.append(Metric(name="expression_variability", value=round(float(np.std(inten)), 5) if len(inten) > 1 else 0.0, unit="coef", method="SD интенсивности по кадрам окна", kind="derived", group="expressivity", status=ResultStatus.ok if len(inten) > 2 else ResultStatus.partial))
        motion = [f["motion"] for f in with_face if f.get("motion") is not None]
        if motion:
            metrics.append(Metric(name="expression_motion", value=round(float(np.mean(motion)), 5), unit="coef/frame", method="средний |Δ| вектора blendshapes между соседними кадрами", kind="derived", group="expressivity"))
        for k in ("yaw", "pitch", "roll", "face_size"):
            vals = [f[k] for f in with_face if f.get(k) is not None]
            if vals:
                metrics.append(Metric(name=f"head_{k}" if k != "face_size" else "face_size", value=round(float(np.mean(vals)), 3), unit="deg" if k != "face_size" else "ratio", method="MediaPipe transformation matrix" if k != "face_size" else "ширина bbox landmarks / ширина кадра", group="pose" if k != "face_size" else "quality"))
        all_keys = sorted({k for f in with_face for k in f.get("blendshapes", {})})
        bs_mean = {k: round(float(np.mean([f["blendshapes"][k] for f in with_face if k in f["blendshapes"]])), 5) for k in all_keys}
        presence = len(with_face) / n
        status = ResultStatus.ok if presence >= 0.5 else ResultStatus.partial
        if presence < 0.5:
            quality.notes.append("лицо обнаружено менее чем в половине кадров")
        if any(f.get("face_size", 1) < 0.08 for f in with_face):
            quality.notes.append("лицо мелкое (<8% ширины кадра)")
        return self.result(ws, we, metrics, status, quality=quality, payload={"blendshapes_mean": bs_mean, "raw": frames})

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        if self._frames and self._win_start is not None:
            return [self._emit_window()]
        return []

    async def shutdown(self) -> None:
        b = getattr(self, "backend", None)
        if b is not None:
            b.close()
