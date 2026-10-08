# Архитектура

Модульный монолит: один процесс FastAPI, asyncio event loop, рабочие потоки для захвата и inference.
Никаких брокеров и микросервисов.

```
Input Source (поток захвата)                 StreamProcessor                       EventRouter
 WindowsLiveSource: DXcam + WASAPI  ─┐   RGB, mono 16 kHz (soxr), чанки 200 мс,     ограниченные очереди
 FileReplaySource:  PyAV/FFmpeg     ─┴─► метки времени, gap/blank/silence, превью ─► на подписчика
                                                                                     │
      ┌──────────────────────────────────────────────────────────────────────────────┤
      ▼ video.frame (drop_oldest)        ▼ audio.chunk (lossless)                     │
   FaceAnalyzer                       VadStage (Silero) ── speech.activity ──┐        │
                                      ProsodyAnalyzer ◄── transcript.final   │        │
                                      SpeechAnalyzer ◄───────────────────────┤        │
                                        ├─ transcript.partial/final          │        │
                                        └─ text.reference (Reading + эталон) │        │
                                      VoiceEmotionAnalyzer ◄─────────────────┘        │
                                      TextAnalyzer, TransactionalAnalyzer (dedupe) ◄──┘
                                                │ analysis.result
                                                ▼
                               Collector → ResultStore → TemporalFusionEngine (1 Гц)
                                         → WebSocket (fan-out)       │ stop
                                                                     ▼
                                     Cross-Modal → ScoringEngine → SQLite + Parquet + summary.json
```

## Время

- `SessionClock`: `session_t0 = time.monotonic_ns()` при старте источника; все события несут `t_ms` от t0.
- Live-видео: метка по монотонным часам в момент захвата.
- Live-аудио: непрерывная ось по счётчику сэмплов, привязанная к часам при первом чанке; `drift_ms` =
  (время прихода − время по счётчику) — диагностика расхождения часов устройств.
- File: media PTS → мс относительно минимального `start_time` потоков; режим realtime синхронизирует доставку с часами.
- Разрыв входного аудио > 40 мс: буфер сбрасывается, следующий чанк несёт `gap_before_ms`; анализаторы сбрасывают буферы.
- Результаты имеют интервалы `[t_start_ms, t_end_ms]`.

## Очереди (EventRouter)

| Политика | Где | Поведение при переполнении |
|---|---|---|
| `drop_oldest` | видео (FaceAnalyzer, размер 2) | выбрасывается старейший кадр, счётчик `dropped` |
| `lossless` | аудио, collector | новый чанк не ставится, длительность явно добавляется в `gap_before_ms` следующего; `lost_audio_ms` |
| `dedupe` | Text/Transactional | событие с уже виденным ключом отбрасывается |

Тяжёлый inference выполняется в `ThreadPoolExecutor`; GPU-модели (Whisper, emotion2vec, LLM) проходят через
`InferenceGate` (семафор `inference.max_parallel`), чтобы не конкурировать за VRAM. Event loop не блокируется.

## Контракт анализатора

`AnalyzerPlugin`: `initialize(config)`, `process(event) → list[AnalysisResult]`, `finalize(session_id)`, `health()`, `shutdown()`;
атрибуты `module_id`, `version`, `subscribes`, `queue_policy`, `queue_size`. `AnalyzerHost` запускает каждый плагин в
отдельной asyncio-задаче: исключение в `process` → `module.error`, счётчик ошибок, обработка продолжается;
`ModelUnavailable` при инициализации → состояние `unavailable`, подписка снимается, остальные модули работают.

Модели подключаются через backend-адаптеры (`analyzers/*/backends.py`): `BACKENDS[name](config)` с методами `load()` и
`transcribe()`/`infer()`; замена модели — новый адаптер + строка в конфиге.

`AnalysisResult` (Pydantic v2): `session_id, module_id, model_id, model_version, t_start_ms, t_end_ms, metrics[], status,
quality, latency_ms, error, finality (provisional|final), payload`. `Metric`: `name, value, unit, normalized_score (0–100|null),
confidence (только если модель даёт интерпретируемую уверенность), method, status, kind (measured|model_output|derived|experimental), group`.
Статусы: `ok, partial, insufficient_data, unavailable, not_applicable, error`.

Большие массивы (кадры, PCM) — только NumPy внутри процесса. Сырые ряды (`payload.raw`: blendshapes по кадрам,
embeddings) уходят в Parquet и вырезаются из JSON/WebSocket.

## Жизненный цикл сессии

`created → initializing → running → stopping → completed | failed`.

- **Start**: идемпотентен. Инициализация модулей (загрузка моделей) → VAD → старт часов и источника.
- **Stop** (или EOF файла): остановка источника и освобождение устройств → flush ресемплера → VAD → анализаторы по
  стадиям (производители: face/speech/voice → потребители: prosody/text/transactional), каждый дочищает очередь и
  вызывает `finalize` → collector → fusion.finalize → cross-modal → scoring → SQLite/Parquet/summary.json.

## Fusion и scoring

`TemporalFusionEngine`: окна 5 с, шаг 1 с; вклад результата — среднее, взвешенное перекрытием; строковые метрики — мода.
Поздние результаты помечают перекрытые окна «грязными» → пересчёт и повторная отправка (`replace_key`). Окно provisional,
пока есть provisional-входы или ожидаются запаздывающие модальности (`pending`), и финализируется по истечении `late_horizon_ms`
или при Stop. `availability` фиксирует для каждой модальности `available | missing | pending | <status>`.

Cross-modal и ScoringEngine описаны в `docs/metrics.md`. `ScoringBackend` — протокол для будущей обученной fusion-модели.

## API

`GET /api/health · /api/devices · /api/modules · /api/config` (`PUT` — в `configs/user.yaml`) · `POST /api/files` ·
`POST /api/sessions` · `POST /api/sessions/{id}/start|stop` · `GET /api/sessions/{id}` · `/results?module&t_from&t_to` ·
`/export?format=json|csv|parquet` · `/preview` · `WS /ws/sessions/{id}`.

WebSocket-сообщения: `session`, `status` (1 Гц: захват, модули, очереди, gate), `preview` (JPEG base64, 2 Гц),
`result`, `transcript.partial|final`, `speech.activity`, `text.reference`, `module_error`.
