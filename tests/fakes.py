"""Тестовые заглушки бэкендов (только тесты; в production-реестр не входят)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.analyzers.speech.alignment import normalize_words
from app.schemas.events import Word


class FakeOllama:
    """Поддельный Ollama HTTP API для httpx.MockTransport.

    mode: ok | invalid_then_ok | always_invalid | timeout | down
    Ответ содержит одну реальную цитату из фрагмента и одну выдуманную (проверка grounding).
    """

    def __init__(self, mode: str = "ok", model: str = "qwen3:8b") -> None:
        import httpx

        self.mode = mode
        self.model = model
        self.chat_calls = 0
        self.prompts: list[str] = []
        self.transport = httpx.MockTransport(self.handle)

    @staticmethod
    def _fragment(user: str) -> str:
        import re

        m = re.search(r"(?:АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ|ФРАГМЕНТ|ОТВЕТ): «(.*)»", user, re.S)
        return m.group(1) if m else ""

    def handle(self, request: Any) -> Any:
        import json

        import httpx

        if self.mode == "down":
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0-fake"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": self.model, "model": self.model, "digest": "fakedigest000"}]})
        assert path == "/api/chat"
        self.chat_calls += 1
        body = json.loads(request.content)
        assert body["options"]["temperature"] == 0 and body["stream"] is False and "format" in body
        user = body["messages"][1]["content"]
        self.prompts.append(user)
        if self.mode == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if self.mode == "always_invalid" or (self.mode == "invalid_then_ok" and self.chat_calls % 2 == 1):
            return httpx.Response(200, json={"message": {"content": "{не json"}})
        frag = self._fragment(user)
        real = " ".join(frag.split()[:2]) or "текст"
        title = body["format"].get("title")
        if title == "TextAnalysisOut":
            out = {
                "sentiment": "positive", "valence": 0.4, "emotion_labels": ["joy"], "emotional_intensity": 0.7,
                "constructions": [{"quote": real, "type": "emotive_lexis", "effect": "усиливает"},
                                  {"quote": "выдуманная цитата", "type": "other", "effect": "-"}],
                "negations_considered": True, "context": "контекст", "source_type": "narration", "explanation_short": "кратко",
            }
        else:
            out = {"status": "ok", "replicas": [
                {"speaker": "толстый", "quote": real, "label": "child_free", "uncertainty": "medium", "rationale": "восклицание",
                 "evidence": [{"marker": "spontaneous_emotion", "quote": real}]},
                {"speaker": "тонкий", "quote": "нет такой", "label": "parent_critical", "uncertainty": "high", "rationale": "-",
                 "evidence": [{"marker": "condemnation", "quote": "абракадабра"}]},
            ], "note": ""}
        return httpx.Response(200, json={"message": {"content": "<think>x</think>" + json.dumps(out, ensure_ascii=False)}, "eval_count": 42})


class FakeSER:
    """Детерминированный «классификатор» голосовой экспрессии: scores зависят от громкости окна."""

    LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unk"]

    def __init__(self, cfg: dict[str, Any], cache_root: str) -> None:
        self.model_id, self.model_version = "fake-ser", "test"
        self.registry = {"model_id": "fake-ser", "version": "test", "backend": "tests"}
        self.calls: list[int] = []

    def load(self) -> None:
        pass

    def infer(self, audio: np.ndarray) -> dict[str, Any]:
        self.calls.append(audio.size)
        s = np.full(9, 0.05)
        s[4] = 0.6
        return {"labels": [f"x/{lab}" for lab in self.LABELS], "scores": (s / s.sum()).tolist(), "embedding": np.arange(16, dtype=np.float32)}


class FakeASR:
    """Детерминированный «ASR»: слова эталона равномерно распределены по [0, total_ms];
    возвращает слова, середина которых попала в переданный фрагмент. Ошибки вносятся
    заменой каждого `corrupt_every`-го слова."""

    reference: str = ""
    total_ms: float = 0.0
    corrupt_every: int = 0

    def __init__(self, cfg: dict[str, Any], root: str, threads: int) -> None:
        self.model_id = "fake-asr"
        self.model_version = "test"
        self.registry = {"model_id": "fake-asr", "version": "test", "backend": "tests"}
        self.words = normalize_words(self.reference)
        self.calls = 0

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, t_offset_ms: float, words: bool, prompt: str | None) -> dict[str, Any]:
        self.calls += 1
        t1 = t_offset_ms + audio.size / 16.0
        step = self.total_ms / max(len(self.words), 1)
        out = []
        for i, w in enumerate(self.words):
            s, e = i * step, (i + 1) * step
            if t_offset_ms <= (s + e) / 2 < t1:
                txt = "ошибка" if self.corrupt_every and i % self.corrupt_every == 0 else w
                out.append(Word(text=txt, t_start_ms=int(s), t_end_ms=int(e - 5), probability=0.9))
        return {"text": " ".join(w.text for w in out), "words": out, "avg_logprob": [-0.2], "no_speech_prob": [0.01],
                "compression_ratio": [1.2], "language_probability": 1.0}
