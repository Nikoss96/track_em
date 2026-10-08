# Метрики

Обозначения `kind`: **measured** — физическое измерение сигнала; **model_output** — выход модели (не установленный факт);
**derived** — детерминированное производное; **experimental** — исследовательская производная без валидации.
`confidence` заполняется только при наличии интерпретируемой уверенности модели — сейчас ни одна модель её не даёт, поле `null`.

## A. Мимика (`face`, окна `window_ms` = 1 с, анализ `analysis_fps`)

Источник: MediaPipe Face Landmarker (52 blendshape-коэффициента 0–1, 478 landmarks, матрица трансформации).
Blendshapes — ARKit-подобные коэффициенты, **не** FACS Action Units.

| Метрика | Формула | Ед. |
|---|---|---|
| `frames_analyzed` | число кадров в окне | кадры |
| `face_presence` | кадры с лицом / кадры окна | 0–1 |
| `brightness`, `sharpness` | средняя яркость серого; дисперсия лапласиана | 0–255; — |
| `smile` | mean(mouthSmileLeft, mouthSmileRight), среднее по кадрам с лицом | coef |
| `brow_raise` / `brow_down` | mean(browInnerUp, browOuterUpL/R) / mean(browDownL/R) | coef |
| `mouth_open` | jawOpen | coef |
| `lip_press`, `lip_stretch`, `lip_pucker`, `lip_funnel`, `lip_corner_down` | соответствующие коэффициенты (L/R усреднены) | coef |
| `eye_blink`, `eye_squint`, `eye_wide`, `cheek_squint`, `nose_sneer` | то же | coef |
| `expression_intensity` | среднее всех blendshapes, кроме `_neutral`, `eyeLook*`, `eyeBlink*` | coef |
| `expression_variability` | SD `expression_intensity` по кадрам окна | coef |
| `expression_motion` | средний \|Δ\| вектора экспрессивных blendshapes между соседними анализируемыми кадрами | coef/кадр |
| `head_yaw/pitch/roll` | углы Эйлера из матрицы трансформации | ° |
| `face_size` | ширина bbox landmarks / ширина кадра | 0–1 |

Статус окна: нет лица → `insufficient_data` (мимические метрики не выдаются); лицо < 50 % кадров → `partial`.
`payload.blendshapes_mean` — средние всех 52 коэффициентов; `raw_face.parquet` — все коэффициенты каждого кадра.
Группа `predicted_facial_expression` зарезервирована для отдельного классификатора эмоций (не подключён).

## B. Интонация (`prosody`)

**Instantaneous** (окно 1 с, шаг 0,5 с): `rms_dbfs` = 20·log10(RMS); `voiced_ratio` — доля 10-мс кадров с F0>0;
`f0_mean_hz` — среднее F0 **только озвученных** кадров (Praat autocorrelation, 75–500 Гц). Тишина (< `silence_dbfs`) →
`insufficient_data`; < 5 озвученных кадров → F0 `insufficient_data`.

**Window-level** (контекст 4 с, шаг 2 с):

| Метрика | Формула |
|---|---|
| `f0_median_hz`, `f0_min_hz`(P5), `f0_max_hz`(P95) | по озвученным кадрам |
| `f0_sd_st` | SD(12·log2(F0/медиана)) — полутоны |
| `f0_range_st` | P90 − P10 в полутонах |
| `speech_ms` | 10-мс кадры с энергией ≥ max(silence_dbfs, P95 − 25 dB) |
| `intensity_mean_db`, `intensity_sd_db`, `intensity_range_db` (P95−P5) | по речевым 10-мс кадрам |
| `pause_count`, `pause_total_ms`, `pause_ratio`, `pause_mean_ms` | паузы: тишина ≥ `pause_min_ms` (250 мс) между участками речи (порог как выше, потоково) |
| `voiced_segment_varco` | 100·SD/mean длительностей озвученных участков ≥ 30 мс — приближение VarcoV без фонемной сегментации |
| `syllable_nuclei_rate` | пики сглаженной энергии (prominence ≥ 2 dB) в озвученных кадрах / секунды озвученной речи (по мотивам de Jong & Wempe, 2009) |
| `egemaps_*` | openSMILE eGeMAPSv02 Functionals (88 признаков в `payload.egemaps`; в метрики вынесены loudness, F0 CV, jitter, shimmer, HNR, voiced segments/s, mean unvoiced) |

**Темп** (по `transcript.final`): `speech_rate_wpm` = слова / длительность фразы × 60 000;
`articulation_rate_wpm` = слова / (длительность − паузы).

## C. Голосовая экспрессия (`voice_emotion`)

Окна `window_ms` (4 с, шаг 2 с) внутри завершённых речевых сегментов VAD; сегмент < `min_speech_ms` → `insufficient_data`.
`score_<label>` — softmax классификационной головы emotion2vec+ (angry, disgusted, fearful, happy, neutral, other, sad, surprised),
**не калиброванная вероятность**; `top_label` — argmax. `positive_minus_negative` (experimental) = happy − (angry+sad+fearful+disgusted).
Каждый результат помечен `validation = not_validated_ru`. Embedding (768/1024) — в `raw_voice_emotion.parquet`.

## D. Речь (`speech`)

Silero VAD (порог 0,5; конец фразы — 600 мс тишины; max 12 с с перекрытием 1 с при разрезе) → faster-whisper (`ru`,
temperature 0, word timestamps). Partial (`finality=provisional`) заменяются финальными по `replace_key`.
Дубли на границах перекрытия удаляются по меткам слов. Метрики: `text`, `word_count`, `avg_logprob`, `no_speech_prob`,
`mean_word_probability` (выходы Whisper, не калиброванная уверенность).

**Reading Mode.** Эталон и гипотеза нормализуются (нижний регистр, ё→е, без пунктуации, дефисы/апострофы внутри слова
склеиваются). Каждая финальная фраза выравнивается полуглобальным DP по словам в окне вокруг текущей позиции чтения
(начало/конец участка эталона свободны). S — замены, D — пропуски (включая ≤ 8 пропущенных слов между фразами),
I — вставки, N — слова эталона во фрагменте. Если стоимость > 0,6·|гипотезы| — `unaligned` (WER не считается).
`wer_segment` = (S+D+I)/N; `reading_accuracy` = 1−WER, `normalized_score` = 100·max(0, 1−WER) (ТЗ §6).
`wer_cumulative` = ΣS+ΣD+ΣI / ΣN только по выровненным завершённым фрагментам — **не** по всему произведению.
`reading_progress` — доля эталона, пройденная указателем.

## E. Текст (`text`)

Единица анализа: Reading + эталон — новые предложения эталона, соответствующие прочитанному фрагменту (каждое один раз),
с контекстом ±; Reading без эталона — распознанный текст; Q&A — ответ участника (≥ `min_chars`) вместе с вопросом.
LLM (Qwen3, temperature 0, JSON Schema) → `sentiment` (positive/negative/neutral/mixed), `valence` [−1, 1],
`emotional_intensity` [0, 1], `emotion_labels`, `source_type`, `payload.constructions` (цитаты, отсутствующие в тексте,
отбрасываются), `negations_considered`, `context`, `explanation_short`. Это оценка **текста**, не исполнителя.

## F. Транзактный анализ (`transactional`)

Метки: `parent_critical, parent_nurturing, adult, child_free, child_adapted, mixed, undetermined`. Маркеры: директивность,
осуждение, покровительство, забота, фактическая аргументация, рациональная структура, эмоциональный протест,
спонтанная эмоция, зависимое/адаптивное реагирование, уступчивость. Каждая метка требует дословных цитат; без них метка
понижается до `undetermined`. Reading: только реплики персонажей; без прямой речи → `not_applicable` (LLM не вызывается).
Q&A: ответ участника; < `min_words` → `insufficient_data`. `primary_label` — мода меток реплик (ничья → `mixed`).
`llm_reported_uncertainty` — самоотчёт LLM, не калиброван. Не является диагностикой эго-состояний.

## Fusion (`fusion`)

Для окна [ws, ws+5 с): `<module>.<metric>` = Σ value·overlap / Σ overlap по результатам со статусом ok/partial;
строки — мода. `payload.availability` — состояние каждой модальности; `payload.alignment` — сопоставленные признаки
лица/голоса/текста и `text_voice_sign` (совпадение знаков валентности текста и `positive_minus_negative`).

## Cross-modal (сессия, описательно)

Не является валидированной шкалой согласованности; расхождение модальностей ≠ ошибка исполнения.
- Ранговые корреляции Спирмена (ρ, p, n окон; n < 8 → `insufficient_data`) для пар: мимика↔F0 SD, мимика↔SD энергии,
  валентность текста↔голос, валентность текста↔улыбка, интенсивность текста↔мимика, интенсивность текста↔F0 SD.
  Окна fusion перекрываются (шаг 1 с при окне 5 с) — p-значения завышают значимость; использовать как описательные.
- Эмоционально значимые фрагменты (`text.emotional_intensity` ≥ 0,6) vs остальные: Cohen's d для интенсивности мимики, F0 SD, SD энергии.
- Паузы и предложения: доля пауз, середина которых в пределах 400 мс от конца предложения (слово ASR с . ! ? …),
  и доля границ предложений, на которые приходится пауза.

## ScoringEngine

Группы: `facial_expressivity, prosodic_expressivity, voice_emotion_profile, reading_accuracy, text_sentiment,
transactional_style, crossmodal_consistency`. Для каждой метрики: `raw` (среднее по финальным результатам), `sd`, `median`,
`n`, `available`, `method`, `normalized` (только при процедуре в `scoring.normalizations`: формула WER или
`{method: linear, min, max, source}`), `normalization_version`.

`exploratory_composite_score` — выключен по умолчанию; Σ wᵢ·normalizedᵢ / Σ wᵢ по метрикам с весами из YAML и
определённой нормализацией; если доступно < `min_available` — `null` с причиной. Всегда `experimental=true, validated=false`.
