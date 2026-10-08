"""Бэкенды лицевого анализа. MediaPipe — основной; Py-Feat — опциональный исследовательский."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.analyzers.base import ModelUnavailable


@dataclass
class FaceFrame:
    face: bool
    blendshapes: dict[str, float] = field(default_factory=dict)
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    face_size: float | None = None  # ширина bbox / ширина кадра
    landmarks: np.ndarray | None = None  # (478, 3) нормализованные


class MediaPipeFaceBackend:
    name = "mediapipe_face_landmarker"

    def __init__(self, model_path: Path, min_presence: float = 0.5) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"mediapipe не импортируется: {e}") from e
        self.mp = mp
        self.version = mp.__version__
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=min_presence,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def __call__(self, rgb: np.ndarray, t_ms: int) -> FaceFrame:
        ts = max(int(t_ms), self._last_ts + 1)  # VIDEO-режим требует строго возрастающих меток
        self._last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ts)
        if not res.face_landmarks:
            return FaceFrame(face=False)
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        lm = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)
        yaw = pitch = roll = None
        if res.facial_transformation_matrixes:
            R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            yaw = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
            roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        size = float(lm[:, 0].max() - lm[:, 0].min())
        return FaceFrame(face=True, blendshapes=bs, yaw=yaw, pitch=pitch, roll=roll, face_size=size, landmarks=lm)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:  # noqa: BLE001
            pass


class PyFeatBackend:
    """Опциональный backend Py-Feat (FACS AU + классификатор эмоций). Требует `pip install py-feat`."""

    name = "py-feat"

    def __init__(self, **_: Any) -> None:
        try:
            import feat  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable("Py-Feat не установлен (опциональный backend, см. docs/models.md)") from e
        raise ModelUnavailable("Py-Feat backend не активирован в этой версии: требуется отдельная валидация AU-моделей")
