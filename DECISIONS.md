# DECISIONS

Краткий журнал архитектурных решений. Одна запись — одно решение.

| # | Решение | Причина |
|---|---------|---------|
| D1 | Профиль по умолчанию `cpu`; профиль `gpu` (RTX 3080) описан в `configs/profiles/gpu.yaml`, выбирается автоматически при наличии CUDA. | На машине разработки нет NVIDIA GPU (Intel Iris Xe, 16 GB RAM). ТЗ ориентировано на RTX 3080 — код поддерживает оба варианта. |
| D2 | Python 3.11 ставится через `uv` в `.venv`; кэши моделей/пакетов (`HF_HOME`, `MODELSCOPE_CACHE`, `UV_CACHE_DIR`) — в `.cache/` проекта. | На C: мало места; 3.11 требуется ТЗ. |
| D3 | Модульный монолит: asyncio event loop + `ThreadPoolExecutor` для inference. Тяжёлые модели проходят через `InferenceGate` (семафор слотов), чтобы не конкурировать за CPU/VRAM. | Нет блокировки event loop; без брокеров/микросервисов. |
| D4 | Время: `session_t0` по `time.monotonic_ns()`; все события несут `t_ms` относительно начала сессии. File Replay использует media PTS → ms; при старте фиксируется offset аудио/видео, drift отслеживается в StreamProcessor. | Единая временная ось. |
| D5 | Внутренние события — `dataclass(slots=True)` с NumPy-массивами; сериализуемые контракты (результаты, статусы, API) — Pydantic v2. | Большие массивы не идут через JSON. |
| D6 | Очереди подписчиков: `drop_oldest` (видео), `lossless` (аудио: при переполнении блок отправителя с таймаутом, затем явная маркировка gap), `dedupe` (NLP). | Требования к целостности аудио. |
| D7 | VAD — Silero VAD (ONNX), поставляемый с faster-whisper. | Без дополнительной зависимости torch для VAD. |
| D8 | LLM-бэкенд — Ollama HTTP `/api/chat` со structured output (`format`=JSON Schema), temperature 0. При недоступности — статус `unavailable`, остальные модули работают. | Требование локальности и замены моделей. |
| D9 | Voice emotion — emotion2vec+ через FunASR (CPU допустим). Применимость к русской речи не валидирована: каждый результат помечается `validation=not_validated_ru`. | Научная корректность. |
| D10 | Storage: sqlite3 (stdlib) для сессий/метаданных/результатов-JSON; Parquet (pyarrow) для временных рядов; экспорт JSON/CSV. | Минимум зависимостей. |
| D12 | COM в MTA (`sys.coinit_flags=0`), SoundCard импортируется до dxcam; патч dxcam 0.3.0 (`capture/dxcam_patch.py`) убирает двойной `Release()`. | Иначе access violation при Stop и ошибка SoundCard `S_FALSE`. |
| D13 | Целевая машина пользователя — RTX 3080 / i9-12900KF / 32 GB: GPU-профиль = whisper large-v3-turbo (int8_float16), emotion2vec+ large, Qwen3-8B; threads 8, max_parallel 3. | Указание пользователя; бюджет VRAM ≈ 9 из 10 ГБ. |
| D14 | Модели ASR/SER/LLM за адаптерами `BACKENDS`; на машине разработки логика проверяется тестовыми бэкендами (FakeASR, FakeSER, поддельный Ollama через httpx.MockTransport), реальные модели — `pytest -m slow` и `scripts/verify_target.py` на целевой машине. | Тяжёлые модели не скачиваются на ноутбук разработки. |
| D15 | NLP-единица в Reading Mode — новые (ещё не анализированные) предложения эталона, покрытые выровненной фразой ASR; каждое предложение анализируется один раз. | Без дублирования LLM-вызовов при перекрытии фраз. |
| D16 | LLM-URL обязан быть локальным (127.0.0.1/localhost), иначе `unavailable`; цитаты-обоснования LLM проверяются на присутствие в тексте, необоснованная транзактная метка понижается до `undetermined`. | Приватность и защита от галлюцинаций. |
| D17 | По запросу пользователя добавлен LLM-бэкенд OpenAI (Chat Completions + Structured Outputs strict, `store:false`, без SDK — httpx). Включается только явным `llm.allow_external`; ключ в env/`configs/secrets.yaml`; вне InferenceGate. Учёт usage/стоимости по тарифам из YAML, лимит `session_budget_usd`. Модель по умолчанию `gpt-4.1-mini` (не reasoning: предсказуемая цена, temperature 0). | Нет локальной LLM у пользователя; ТЗ требовало локальность — отступление осознанное и опциональное. |
| D18 | File Replay без реального времени — back-pressure: источник ждёт освобождения очередей анализаторов вместо выбрасывания кадров. | Иначе результаты ускоренного офлайн-анализа недетерминированы (найдено тестом воспроизводимости). |
| D11 | Frontend: React 18 + Vite 5 + Recharts (Node 18 на машине; Vite 6+/7 требуют Node 20+). | Совместимость окружения. |
