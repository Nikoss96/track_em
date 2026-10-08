# Модели

Все веса — открытые; загружаются при установке (`scripts/download_models.py`) или первом запуске модуля в `.cache/`.
Фактические версии и sha256 каждой сессии фиксируются в реестре моделей (SQLite, экран «Настройки»).
Лицензии весов требуют проверки перед публикацией результатов — ниже указаны сведения из карточек моделей на момент интеграции.

| Модуль | Модель (профиль) | Источник | Лицензия | Языки / применимость | Память |
|---|---|---|---|---|---|
| Мимика | MediaPipe Face Landmarker `float16/1` | storage.googleapis.com/mediapipe-models | Apache-2.0 | n/a | CPU, ≈ 4 МБ, ~10–30 мс/кадр |
| VAD | Silero VAD v6 (ONNX, в составе faster-whisper) | github.com/snakers4/silero-vad | MIT | многоязычный | CPU, < 5 МБ |
| ASR | faster-whisper `small` (cpu) / `large-v3-turbo` (gpu, int8_float16) | huggingface.co/Systran, mobiuslabsgmbh | MIT | многоязычный, `ru` | small int8 CPU ≈ 0,5 ГБ RAM; turbo ≈ 1,6 ГБ VRAM |
| Голос | emotion2vec+ `base` (cpu) / `large` (gpu) через FunASR | huggingface.co/emotion2vec | см. карточку (исследовательское использование) | обучение преимущественно zh/en; **ru не валидирован** | base ≈ 0,4 ГБ; large ≈ 1,3 ГБ VRAM |
| Просодия | Praat (Parselmouth 0.4.7), openSMILE 2.6 eGeMAPSv02, librosa | — | GPL-3 (Praat/Parselmouth), openSMILE: исследовательская некоммерческая | n/a | CPU |
| Текст / ТА | Qwen3 `4b` (cpu) / `8b` (gpu, Q4_K_M) через Ollama | ollama.com/library/qwen3 | Apache-2.0 | многоязычная, русский поддерживается | 8b ≈ 6 ГБ VRAM при ctx 4096 |
| Текст / ТА (альтернатива) | OpenAI API: `gpt-4.1-mini` по умолчанию (настраивается) | api.openai.com | OpenAI API Terms; **внешний сервис** | multilingual (ru) | — (облако); Structured Outputs strict; `store:false` |
| (опц.) | Py-Feat (FACS AU + эмоции) | github.com/cosanlab/py-feat | MIT (код), веса — разные | n/a | не активирован: требует валидации AU-моделей |
| (опц.) | WhisperX forced alignment | github.com/m-bain/whisperX | BSD-2 | ru через wav2vec2-модели | не подключён (план: финальный офлайн-анализ) |

## Бюджет VRAM на RTX 3080 (10 ГБ)

Whisper large-v3-turbo int8_float16 ≈ 1,6 ГБ + emotion2vec+ large ≈ 1,3 ГБ + Qwen3-8B Q4_K_M ≈ 6 ГБ ≈ 9 ГБ.
Модели загружаются при старте сессии; одновременные GPU-вызовы ограничены `InferenceGate` (`inference.max_parallel`).
Фактическое потребление: `nvidia-smi` в шапке UI и `scripts/verify_target.py` (пик VRAM). При нехватке: `qwen3:4b`
(≈ 3 ГБ) или `emotion2vec_plus_base`. MediaPipe и Praat/openSMILE работают на CPU.

## Применимость к русскому языку

- **Whisper**: русский входит в обучающие данные; качество на конкретном материале оценивается по WER относительно эталона
  (Reading Mode) — это и есть встроенная валидация ASR.
- **emotion2vec+**: валидация на русской речи не проводилась. Каждый результат помечен `not_validated_ru`.
  Для валидации нужен размеченный русский корпус (например, RESD / Dusha) и сравнение с разметкой — см. `experiments/`.
- **Qwen3**: русский поддерживается; оценки сентимента/транзактных меток не валидированы относительно экспертной разметки.

## Замена модели

1. Реализовать адаптер в `backend/app/analyzers/<module>/backends.py` (методы `load()` и `infer()`/`transcribe()`,
   поля `model_id`, `model_version`, `registry`).
2. Добавить его в `BACKENDS` и указать `modules.<module>.backend` в `configs/*.yaml` или в «Настройках».
3. Контракт результата и UI не меняются; версия модели попадает в каждый результат и в реестр.
