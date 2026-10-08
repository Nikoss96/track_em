"""Реестр анализаторов. Импорт ленивый: ошибка импорта одного модуля не ломает остальные."""

from __future__ import annotations

import importlib
from typing import Any

MODULES: dict[str, dict[str, Any]] = {
    "face": {"path": "app.analyzers.face.analyzer:FaceAnalyzer", "title": "Лицевая экспрессия", "stage": 0},
    "speech": {"path": "app.analyzers.speech.analyzer:SpeechAnalyzer", "title": "Распознавание речи", "stage": 0},
    "voice_emotion": {"path": "app.analyzers.voice_emotion.analyzer:VoiceEmotionAnalyzer", "title": "Голосовая экспрессия", "stage": 0},
    "prosody": {"path": "app.analyzers.prosody.analyzer:ProsodyAnalyzer", "title": "Интонация (просодия)", "stage": 1},
    "text": {"path": "app.analyzers.text.analyzer:TextAnalyzer", "title": "Семантика текста", "stage": 1},
    "transactional": {"path": "app.analyzers.transactional.analyzer:TransactionalAnalyzer", "title": "Транзактный анализ", "stage": 1},
    # только для тестов инфраструктуры
    "mock": {"path": "app.analyzers.mock.analyzer:MockAnalyzer", "title": "Mock (тесты)", "stage": 0, "test_only": True},
}

PRODUCTION_MODULES = [k for k, v in MODULES.items() if not v.get("test_only")]


def load_class(module_id: str) -> type:
    spec = MODULES[module_id]["path"]
    mod_name, cls_name = spec.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)
