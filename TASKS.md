# TASKS

Состояние: `[x]` готово и проверено здесь, `[T]` реализовано, проверка — на целевой машине (RTX 3080), `[ ]` не сделано.
Последнее обновление: 2026-10-08.

## Как продолжить / проверить
- Быстрые тесты: `.venv\Scripts\python.exe -m pytest tests -m "not slow"` (37 PASS).
- На целевой машине: `scripts\install.ps1 -Gpu -Models`, `ollama pull qwen3:8b`, затем
  `pytest tests -m slow` и `scripts\verify_target.py` → `data/verification_report.json`.
- Тяжёлые модели (Whisper, emotion2vec, Qwen3) на машине разработки **не скачивались и не запускались**.

## Phase 1 — Foundation
- [x] Структура, `.venv` (Python 3.11.17, uv), `pyproject.toml`, `requirements.txt` / `requirements.lock`, ruff чистый
- [x] Pydantic-контракты, события, AnalyzerPlugin/Host, InferenceGate, EventRouter, SessionManager
- [x] FastAPI + WebSocket, React UI

## Phase 2 — Media Streaming
- [x] FileReplaySource; 100 с в реальном времени без аварии, RSS стабилен (382→407 МБ)
- [x] WindowsLiveSource — проверен на этой машине: smoke PASS и полная Live-сессия через API (1280×720 @30, loopback 48 кГц стерео)
- [x] StreamProcessor, очереди, VAD (Silero), патч dxcam (D12)

## Phase 3 — Core Analyzers
- [x] FaceAnalyzer (MediaPipe) — реальная модель, тесты PASS
- [x] ProsodyAnalyzer (Praat, eGeMAPS, паузы, ритм, темп) — тесты PASS
- [x] SpeechAnalyzer — логика (сегменты, overlap-дедуп, WER, text.reference) PASS с FakeASR
- [T] SpeechAnalyzer с faster-whisper — `tests/integration/test_speech.py` (slow)
- [x] VoiceEmotionAnalyzer — логика PASS с FakeSER (найден и исправлен баг непокрытого хвоста сегмента)
- [T] emotion2vec+ через FunASR — `tests/integration/test_voice_emotion.py` (slow)
- [ ] Py-Feat backend (сейчас честный `unavailable`), WhisperX forced alignment (офлайн) — не реализованы

## Phase 4 — NLP
- [x] Ollama-адаптер (JSON Schema, temperature 0, таймаут, retry, локальный URL), TextAnalyzer, TransactionalAnalyzer
- [x] Reading/Q&A, проверка цитат, not_applicable/insufficient_data, работа без LLM — PASS с поддельным Ollama
- [T] Реальный Qwen3 через Ollama (качество промптов на русском текстах)

- [x] OpenAI-бэкенд (D17): structured outputs, учёт токенов/стоимости, бюджет сессии, ретраи 429/5xx, согласие на внешний сервис, ключ в UI — тесты с поддельным API PASS
- [T] Реальный OpenAI API (нужен ключ пользователя); оценка стоимости — `scripts/estimate_llm_cost.py`
- [x] Back-pressure для ускоренного File Replay (D18) — воспроизводимость 3/3

## Phase 5 — Fusion & Scoring
- [x] TemporalFusionEngine (перекрытие, поздние результаты → пересчёт, provisional→final), Cross-Modal, ScoringEngine, composite
- [x] Полная сессия всех 6 модулей end-to-end (`test_full_session.py`)

## Phase 6 — Research Dashboard
- [x] Setup / Live / Results / Settings; проверено headless-скриншотами на реальной сессии
- [x] Экспорт JSON/CSV/Parquet, фильтры, сравнение эталона, ошибки/пропуски/версии моделей, VRAM в шапке

## Phase 7 — Verification
- [x] 60+ с File Replay; Live start/stop на Windows; синхронизация (непрерывная ось, тесты); отключение модуля;
      работа без LLM; отсутствие лица; тишина; отсутствие дублей ASR; освобождение ресурсов; JSON/CSV; RAM;
      недоступная модель; воспроизводимость (face+prosody идентичны)
- [T] VRAM, задержки и воспроизводимость с реальными GPU-моделями — `scripts/verify_target.py`
- [x] README, docs/architecture.md, docs/metrics.md, docs/models.md

## Идеи следующего этапа
- Валидация emotion2vec на русском корпусе (RESD/Dusha) в `experiments/`.
- WhisperX forced alignment для финального офлайн-анализа; Py-Feat AU как второй лицевой backend.
- Обученная fusion-модель вместо RuleBasedScoring (протокол `ScoringBackend`).

## Известные проблемы
- При первой установке часть пакетов попала в системный Python 3.12 (`C:\Python312`); проект их не использует.
- Fixture `reading_av.mp4`: после ~22 с zoom обрезает лицо (face_presence → 0) — артефакт фикстуры.
- p-значения корреляций по перекрывающимся окнам fusion завышают значимость (отмечено в docs/metrics.md).
