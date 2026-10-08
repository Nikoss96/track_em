"""ASR-бэкенды. Контракт: load() → None; transcribe(audio16k, t_offset_ms, words, prompt) → dict."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from app.analyzers.base import ModelUnavailable
from app.config import cuda_available
from app.schemas.events import Word


class ASRBackend(Protocol):
    model_id: str
    model_version: str
    registry: dict[str, Any]

    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, t_offset_ms: float, words: bool, prompt: str | None) -> dict[str, Any]: ...


class FasterWhisperBackend:
    def __init__(self, cfg: dict[str, Any], models_root: str, threads: int) -> None:
        try:
            import faster_whisper  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"faster-whisper недоступен: {e}") from e
        self.cfg = cfg
        self.name = cfg.get("model", "small")
        device = cfg.get("device", "auto")
        self.device = ("cuda" if cuda_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not cuda_available():
            raise ModelUnavailable("в конфигурации device=cuda, но CUDA недоступна (проверьте драйвер/cuBLAS/cuDNN)")
        ct = cfg.get("compute_type", "auto")
        self.compute_type = ("float16" if self.device == "cuda" else "int8") if ct == "auto" else ct
        self.root = models_root
        self.threads = threads
        self.language = cfg.get("language", "ru")
        self.beam = int(cfg.get("beam_size", 1))
        import faster_whisper

        self.model_id = f"faster-whisper/{self.name}"
        self.model_version = f"faster-whisper {faster_whisper.__version__}; {self.device}/{self.compute_type}"
        self.registry = {
            "model_id": f"faster-whisper-{self.name}", "version": self.compute_type,
            "backend": f"faster-whisper {faster_whisper.__version__} ({self.device})",
            "source": "https://huggingface.co/Systran (faster-whisper CT2 conversions)", "license": "MIT",
            "languages": "multilingual (ru)", "local_path": models_root, "notes": "CTranslate2-конвертация OpenAI Whisper",
        }

    def load(self) -> None:
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(self.name, device=self.device, compute_type=self.compute_type, download_root=self.root, cpu_threads=self.threads)
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"не удалось загрузить whisper '{self.name}' ({self.device}/{self.compute_type}): {e}") from e

    def transcribe(self, audio: np.ndarray, t_offset_ms: float, words: bool, prompt: str | None) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam, word_timestamps=words, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=prompt, temperature=0.0,
        )
        out_words: list[Word] = []
        texts, lp, nsp, cr = [], [], [], []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            texts.append(s.text.strip())
            lp.append(s.avg_logprob)
            nsp.append(s.no_speech_prob)
            cr.append(s.compression_ratio)
            for w in s.words or []:
                out_words.append(Word(text=w.word.strip(), t_start_ms=int(t_offset_ms + w.start * 1000), t_end_ms=int(t_offset_ms + w.end * 1000), probability=float(w.probability)))
        return {"text": " ".join(t for t in texts if t), "words": out_words, "avg_logprob": lp, "no_speech_prob": nsp,
                "compression_ratio": cr, "language_probability": float(info.language_probability)}


BACKENDS: dict[str, Any] = {"faster_whisper": FasterWhisperBackend}
