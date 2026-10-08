"""Бэкенды голосовой экспрессии. Контракт: load(); infer(audio16k) → {labels, scores, embedding}."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.analyzers.base import ModelUnavailable

log = logging.getLogger(__name__)


class Emotion2VecBackend:
    """emotion2vec+ (base/large) через FunASR. Классы: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unk.

    Scores — softmax классификационной головы; это выход модели, а не калиброванная вероятность
    эмоционального состояния. Применимость к русской речи не валидирована.
    """

    LICENSE = "веса: см. карточку модели emotion2vec (исследовательское использование)"

    def __init__(self, cfg: dict[str, Any], cache_root: str) -> None:
        try:
            import funasr  # noqa: F401
            import torch
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"funasr/torch недоступны: {e}") from e
        self.cfg = cfg
        self.model_name = cfg.get("model", "emotion2vec/emotion2vec_plus_base")
        self.hub = cfg.get("hub", "hf")
        dev = cfg.get("device", "cpu")
        if dev == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("device=cuda, но torch.cuda недоступен (установлена CPU-сборка torch? см. README)")
        self.device = dev
        self.cache_root = cache_root
        import funasr

        self.model_id = self.model_name
        self.model_version = f"funasr {funasr.__version__}; torch {torch.__version__}; {dev}"
        self.registry = {
            "model_id": self.model_name, "version": "hub:" + self.hub, "backend": self.model_version,
            "source": f"https://huggingface.co/{self.model_name}" if self.hub == "hf" else f"https://modelscope.cn/models/{self.model_name}",
            "license": self.LICENSE, "languages": "обучение преимущественно zh/en; ru не валидирован",
            "local_path": cache_root, "notes": "9 классов; scores не калиброваны",
        }

    def load(self) -> None:
        from funasr import AutoModel

        try:
            self.model = AutoModel(model=self.model_name, hub=self.hub, device=self.device, disable_update=True, disable_pbar=True, log_level="ERROR")
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"не удалось загрузить {self.model_name}: {e}") from e

    def infer(self, audio: np.ndarray) -> dict[str, Any]:
        res = self.model.generate(audio.astype(np.float32), fs=16000, granularity="utterance", extract_embedding=True, disable_pbar=True)
        r = res[0]
        labels = [str(lab).split("/")[-1].strip("<>") for lab in r["labels"]]
        scores = [float(s) for s in r["scores"]]
        emb = r.get("feats")
        emb = np.asarray(emb, dtype=np.float32).reshape(-1) if emb is not None else None
        return {"labels": labels, "scores": scores, "embedding": emb}


BACKENDS: dict[str, Any] = {"emotion2vec": Emotion2VecBackend}
