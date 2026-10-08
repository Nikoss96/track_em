# Multimodal Behavior Analyzer

Локальное исследовательское приложение для анализа эмоциональной экспрессии и коммуникативного поведения
человека, читающего художественный текст (Reading Mode) или отвечающего на вопрос (Q&A Mode), в реальном времени.
Источник — область экрана Windows + системный звук (loopback) или видеофайл.

Независимые анализаторы: мимика (MediaPipe), интонация (Praat/openSMILE), голосовая экспрессия (emotion2vec+),
речь (faster-whisper + сопоставление с эталоном), семантика текста и транзактный анализ (локальная LLM Qwen3 через Ollama).
Результаты объединяются на общей временной оси (late fusion), сохраняются (SQLite/Parquet) и экспортируются (JSON/CSV).

> **Ограничения интерпретации.** Приложение измеряет наблюдаемые сигналы и выдаёт выходы моделей. Оно **не** устанавливает
> внутреннее эмоциональное состояние, искренность, ложь или личностные характеристики. Сентимент текста — свойство
> текста, а не чтеца. Транзактные метки — наблюдаемые речевые паттерны, а не диагностика эго-состояний.
> Модель голосовой экспрессии на русской речи не валидирована. Индекс `exploratory_composite_score` — экспериментальный
> и невалидированный (выключен по умолчанию). Точность модулей не заявляется без тестового датасета.

## Системные требования

| | Минимум (CPU-профиль) | Целевая конфигурация (GPU-профиль) |
|---|---|---|
| ОС | Windows 10/11 x64 | Windows 10/11 x64 |
| CPU | 4 ядра | Intel i9-12900KF |
| RAM | 16 GB | 32 GB |
| GPU | — | NVIDIA RTX 3080 10 GB, драйвер ≥ 570 (CUDA 12.8) |
| Диск | 10 GB | 20 GB (модели ≈ 8 GB) |
| ПО | Python 3.11 (ставится автоматически через uv), Node.js ≥ 18, FFmpeg в PATH, Ollama (для NLP) | то же |

Профиль выбирается автоматически (`profile: auto` → `gpu`, если CTranslate2 видит CUDA). Бюджет VRAM — в `configs/profiles/gpu.yaml`.

## Установка (Windows)

```powershell
# 1. FFmpeg и Node.js (если нет)
winget install Gyan.FFmpeg OpenJS.NodeJS.LTS
# 2. Ollama (локальная LLM) — https://ollama.com/download , затем:
ollama pull qwen3:8b          # GPU-профиль; для CPU: qwen3:4b
# 3. Проект: Python 3.11 + зависимости + сборка UI + фикстуры
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Gpu -Models
```

`-Gpu` ставит torch CUDA 12.8 и DLL cuBLAS/cuDNN для faster-whisper; `-Models` заранее скачивает веса
(Whisper large-v3-turbo, emotion2vec+ large, MediaPipe Face Landmarker) — после этого сеть не нужна.
Все кэши моделей и пакетов лежат в `.cache/` проекта.

### Проверка CUDA и окружения

```powershell
.venv\Scripts\python.exe scripts\check_env.py
```
Проверяет Python, CTranslate2 CUDA и доступные compute types, torch CUDA, `nvidia-smi` (VRAM), FFmpeg, Ollama и модель,
мониторы (DXcam), аудиовыходы (loopback), диск и RAM.

## Запуск

```powershell
powershell -File scripts\run_backend.ps1        # API + UI: http://127.0.0.1:8000
powershell -File scripts\run_frontend_dev.ps1   # (разработка UI) http://localhost:5173
```

Экраны UI: **Сессия** (настройка и Start/Stop/Reset), **Live Dashboard**, **Результаты**, **Настройки**.
Прямые ссылки: `#/live/<id>`, `#/results/<id>`.

### File Replay

В «Сессии» выберите источник *File Replay*, загрузите MP4/MKV/WAV (или укажите локальный путь), для Reading Mode —
эталонный TXT. Флаг «реальная скорость» воспроизводит файл темпом записи; без него — максимально быстро (для пакетного анализа).
Из командной строки:
```powershell
.venv\Scripts\python.exe scripts\run_session.py --file D:\rec\reading.mp4 --reference D:\rec\text.txt
```

### Live Capture (экран + системный звук)

1. Откройте окно с видео чтеца (видеосвязь, плеер).
2. «Сессия» → источник *Live*, монитор, область (снимите «весь экран» и задайте left/top/width/height —
   координаты в пикселях монитора), аудиоустройство вывода, через которое звучит голос.
3. Start. Захват начинается только после Start и полностью освобождает устройства после Stop.

Системный звук захватывается через WASAPI loopback выбранного **выходного** устройства: звук должен воспроизводиться
через это устройство (громкость системы на результат не влияет, но mute приложения — влияет).
Smoke-тест устройств: `.venv\Scripts\python.exe scripts\smoke_live.py --seconds 5 --region 0,0,1280,720`.

### Проверка на целевой машине

```powershell
.venv\Scripts\python.exe -m pytest tests -m "not slow"     # быстрые тесты (без тяжёлых моделей)
.venv\Scripts\python.exe -m pytest tests -m slow           # реальные модели: MediaPipe, Praat, Whisper, emotion2vec
.venv\Scripts\python.exe scripts\verify_target.py          # полная сессия + Live + пики RAM/VRAM + воспроизводимость
```
Отчёт — `data/verification_report.json`.

## Текстовая аналитика через OpenAI API (вместо локальной LLM)

Модули «Семантика текста» и «Транзактный анализ» могут работать через OpenAI API вместо Ollama.

1. «Настройки» → «Текст / транзактный анализ»: *Бэкенд LLM* = `openai`, включить
   «Разрешить отправку текста во внешний сервис», выбрать модель (по умолчанию `gpt-4.1-mini`), лимит $ на сессию → «Сохранить».
2. «Настройки» → «OpenAI API-ключ»: вставить ключ `sk-…` (хранится в `configs/secrets.yaml`, не возвращается через API)
   или задать переменную окружения `OPENAI_API_KEY`.
3. Новые сессии используют OpenAI; модель и стоимость видны в Live (строка LLM) и в «Результатах».

**Что уходит в OpenAI:** только текст — фрагменты эталона (Reading), распознанные ответы и вопрос (Q&A).
Аудио, видео и метрики не отправляются. Запросы идут с `store: false`. Для исследований с участниками это нужно
отразить в информированном согласии. Без флага `allow_external` бэкенд не запускается (`unavailable`).

**Стоимость** (оценка `scripts/estimate_llm_cost.py`, тарифы на 2026-10-08, 130 слов/мин; фактическая стоимость считается
по usage каждого ответа и показывается в UI; лимит — `llm.session_budget_usd`, по умолчанию $2):

| Модель | Reading, $/час записи | Q&A, $/час |
|---|---|---|
| gpt-4.1-nano | ≈ 0.13 | ≈ 0.06 |
| gpt-4o-mini | ≈ 0.20 | ≈ 0.09 |
| **gpt-4.1-mini** (по умолчанию) | **≈ 0.53** | **≈ 0.24** |
| gpt-5-mini (+ reasoning-токены) | ≈ 0.49 + 0.1…1.0 | ≈ 0.23 + … |
| gpt-4.1 | ≈ 2.6 | ≈ 1.2 |

Для текста с большим количеством коротких диалоговых реплик число вызовов растёт (до ≈ ×2,5).
Пересчёт под свой текст: `.venv\Scripts\python.exe scripts\estimate_llm_cost.py --reference мой_текст.txt --wpm 120`.
Тарифы — в `configs/default.yaml` → `llm.pricing` (при изменении цен OpenAI обновите их).

## Метрики (кратко; формулы — `docs/metrics.md`)

| Модуль | Измерения (measured) | Выходы моделей (model_output) |
|---|---|---|
| Мимика | качество кадра, доля кадров с лицом, поза головы | 52 blendshapes MediaPipe и группы (улыбка, брови, губы, рот), интенсивность/изменчивость экспрессии |
| Интонация | F0 (только озвученные кадры), SD/диапазон F0 в полутонах, энергия, паузы, ритм, темп речи, eGeMAPS | — |
| Голос | — | scores 8 классов emotion2vec+, top-label, embedding |
| Речь | WER/S/D/I на выровненных фрагментах, прогресс чтения | транскрипция, временные метки слов, log-prob |
| Текст | — | сентимент, валентность, эмоции, интенсивность, конструкции (с проверкой цитат) |
| Транзактный | — | метки Р/В/Д по репликам с цитатами-обоснованиями |
| Fusion | окна 5 с/шаг 1 с, доступность модальностей, описательные корреляции | — |

Нормализация 0–100 применяется только по явной процедуре (сейчас: `reading_accuracy = 100·max(0, 1−WER)`).

## Хранение и приватность

- `data/mba.sqlite3` — сессии, конфигурации, статусы, результаты, реестр моделей (версии, sha256).
- `data/sessions/<id>/` — `metrics.parquet` (длинный формат), `raw_face.parquet` (все blendshapes по кадрам),
  `raw_voice_emotion.parquet` (embeddings), `summary.json`; при включённой опции — исходная медиазапись Live.
- Экспорт: `GET /api/sessions/<id>/export?format=json|csv|parquet`.
- Политика хранения: `storage.retention_days` (0 — бессрочно).
- По умолчанию внешние сервисы не используются (Ollama — только локальный URL). Исключение — OpenAI-бэкенд,
  включаемый явно (`llm.allow_external`), см. раздел выше.

## Типичные ошибки

| Симптом | Причина / решение |
|---|---|
| `speech: unavailable … cublas64_12.dll` | не установлены CUDA DLL: `install.ps1 -Gpu` или `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` |
| `voice_emotion: … torch.cuda недоступен` | установлена CPU-сборка torch: переустановите с `--index-url https://download.pytorch.org/whl/cu128` или `device: cpu` |
| `text/transactional: unavailable — Ollama не отвечает` | запустите Ollama; `ollama pull qwen3:8b`; остальные модули работают |
| `модель … не загружена в Ollama` | `ollama pull <llm.model>` или измените модель в «Настройках» |
| Live: «нет сигнала» видео | область вне монитора, защищённый контент (DRM) или свернутое окно |
| Live: тишина | звук идёт через другое устройство вывода — выберите его в «Сессии» |
| `text: unavailable — OpenAI — внешний сервис` | включите `llm.allow_external` в «Настройках» |
| `OpenAI: неверный API-ключ (401)` / `нет ключа` | проверьте ключ в «Настройках» или `OPENAI_API_KEY` |
| `insufficient_quota` | пополните баланс аккаунта OpenAI |
| `бюджет LLM на сессию исчерпан` | увеличьте `llm.session_budget_usd` (0 — без лимита) |
| Нехватка VRAM | `llm.model: qwen3:4b`, `speech.compute_type: int8_float16`, `voice_emotion.model: …plus_base` |
| Отставание ASR | увеличьте `partial_interval_ms` или выключите partial (0); задержки видны в таблице модулей |

## Структура

```
backend/app/  capture · streaming · preprocessing · analyzers/{face,prosody,voice_emotion,speech,text,transactional}
              fusion · scoring · storage · session · api · schemas
frontend/     React + TypeScript + Vite 5 + Recharts
configs/      default.yaml, profiles/{cpu,gpu}.yaml, user.yaml (из UI)
tests/        unit, integration, fixtures;  scripts/ установка, проверки, фикстуры;  docs/
```
Состояние работ — `TASKS.md`, архитектурные решения — `DECISIONS.md`.
