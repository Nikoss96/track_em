"""Module D: SpeechAnalyzer — faster-whisper поверх сегментов Silero VAD.

- аудиобуфер с перекрытием; provisional (transcript.partial) и finalized (transcript.final) текст;
- принудительно разрезанные длинные сегменты транскрибируются с перекрытием 1 с,
  дубли на границе удаляются по временным меткам слов;
- Reading Mode: последовательное выравнивание с эталоном, S/D/I, WER по завершённым фрагментам,
  публикация text.reference для текущего фрагмента произведения.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from app.analyzers.base import AnalyzerPlugin, ModelUnavailable
from app.analyzers.speech.alignment import ReadingTracker
from app.analyzers.speech.backends import BACKENDS
from app.config import models_dir
from app.schemas.events import AudioChunk, Event, EventType, SpeechActivity, TextReference, Transcript, Word
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus

M = ResultStatus
OVERLAP_MS = 1000
PAD_MS = 200
_HALLUCINATIONS = re.compile(r"(субтитр|продолжение следует|спасибо за просмотр|редактор субтитров|подписывайтесь)", re.I)


class SpeechAnalyzer(AnalyzerPlugin):
    module_id = "speech"
    version = "0.1.0"
    subscribes = (EventType.audio_chunk, EventType.speech_activity)
    queue_policy = "lossless"
    queue_size = 3000

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        bname = config.get("backend", "faster_whisper")
        if bname not in BACKENDS:
            raise ModelUnavailable(f"неизвестный ASR backend: {bname}")
        root = str(models_dir(self.ctx.config) / "whisper")
        self.backend = BACKENDS[bname](config, root, int(self.ctx.config["inference"].get("threads", 4)))
        await self.run_blocking(self.backend.load, gated=True)
        self.model_id = self.backend.model_id
        self.model_version = self.backend.model_version
        self.ctx.shared["register_model"](**self.backend.registry)
        self.language = config.get("language", "ru")
        self.beam = int(config.get("beam_size", 1))
        self.word_ts = bool(config.get("word_timestamps", True))
        self.partial_ms = float(config.get("partial_interval_ms", 2000))
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0: float | None = None
        self.seg_start: float | None = None
        self.seg_id = 0
        self.next_overlap_from: float | None = None
        self.last_word_end = -1.0
        self.last_partial = 0.0
        self.history = ""
        self.tracker = ReadingTracker(self.ctx.reference_text) if (self.ctx.mode == "reading" and self.ctx.reference_text) else None
        self.finals = 0

    # --- буфер ------------------------------------------------------------
    def _append(self, ch: AudioChunk) -> None:
        if self.buf_t0 is None or ch.gap_before_ms > 0:
            self.buf = ch.pcm16k.copy()
            self.buf_t0 = float(ch.t_ms)
        else:
            self.buf = np.concatenate([self.buf, ch.pcm16k])
        keep_from = (self.seg_start - OVERLAP_MS - PAD_MS) if self.seg_start is not None else self.buf_end - 3000
        drop = int((keep_from - self.buf_t0) * 16)
        if drop > 0:
            self.buf = self.buf[drop:]
            self.buf_t0 += drop / 16.0
        max_len = 60 * 16000  # предохранитель
        if self.buf.size > max_len:
            self.buf_t0 += (self.buf.size - max_len) / 16.0
            self.buf = self.buf[-max_len:]

    @property
    def buf_end(self) -> float:
        return (self.buf_t0 or 0.0) + self.buf.size / 16.0

    def _slice(self, t0: float, t1: float) -> tuple[np.ndarray, float]:
        assert self.buf_t0 is not None
        t0 = max(t0, self.buf_t0)
        a = int((t0 - self.buf_t0) * 16)
        b = int((t1 - self.buf_t0) * 16)
        return self.buf[a:max(a, b)], t0

    # --- обработка --------------------------------------------------------
    async def process(self, event: Event) -> list[AnalysisResult]:
        if isinstance(event, AudioChunk):
            self._append(event)
            if self.seg_start is not None and self.partial_ms > 0:
                now = time.monotonic()
                if (now - self.last_partial) * 1000 >= self.partial_ms and self.ctx.gate.waiting == 0 and self.buf_end - self.seg_start > 1500:
                    self.last_partial = now
                    return await self._partial()
            return []
        assert isinstance(event, SpeechActivity)
        if event.kind == "start":
            self.seg_start = float(event.t_ms)
            self.seg_id = event.segment_id
            self.last_partial = time.monotonic()
            return []
        if event.kind == "cancel":
            self.seg_start = None
            return []
        # end
        self.seg_start = None
        start = float(event.t_ms)
        if self.next_overlap_from is not None and self.next_overlap_from < start:
            start = self.next_overlap_from
        self.next_overlap_from = float(event.t_end_ms) - OVERLAP_MS if event.forced else None
        return await self._final(start, float(event.t_end_ms), event.segment_id, float(event.t_ms))

    def _transcribe(self, audio: np.ndarray, t_offset: float, words: bool) -> dict[str, Any]:
        # контекст — только собственная предыдущая транскрипция (не эталон, чтобы не завышать точность)
        return self.backend.transcribe(audio, t_offset, words, self.history[-200:] or None)

    async def _partial(self) -> list[AnalysisResult]:
        assert self.seg_start is not None
        audio, t0 = self._slice(self.seg_start - PAD_MS, self.buf_end)
        if audio.size < 16000:
            return []
        t = time.perf_counter()
        r = await self.run_blocking(self._transcribe, audio, t0, False)
        if not r["text"]:
            return []
        self.ctx.publish(Transcript(type=EventType.transcript_partial, session_id=self.ctx.session_id, t_ms=int(self.seg_start), t_end_ms=int(self.buf_end), text=r["text"], segment_id=self.seg_id, language=self.language))
        return [self.result(self.seg_start, self.buf_end, [Metric(name="text", value=r["text"], method="faster-whisper (provisional)", kind="model_output")],
                            finality="provisional", latency_ms=round((time.perf_counter() - t) * 1000, 1),
                            payload={"replace_key": f"seg{self.seg_id}", "text": r["text"], "segment_id": self.seg_id})]

    async def _final(self, start: float, end: float, seg_id: int, vad_start: float) -> list[AnalysisResult]:
        audio, t0 = self._slice(start - PAD_MS, end + PAD_MS)
        if audio.size < 0.3 * 16000:
            return [self.result(vad_start, end, [], M.insufficient_data, payload={"replace_key": f"seg{seg_id}", "segment_id": seg_id, "reason": "слишком короткий сегмент"})]
        t = time.perf_counter()
        r = await self.run_blocking(self._transcribe, audio, t0, self.word_ts)
        lat = round((time.perf_counter() - t) * 1000, 1)
        words: list[Word] = r["words"]
        # дедупликация на границе перекрытия: слова, закончившиеся до конца уже финализированных
        dropped = [w for w in words if w.t_end_ms <= self.last_word_end + 50]
        words = [w for w in words if w.t_end_ms > self.last_word_end + 50]
        if self.word_ts:
            text = " ".join(w.text for w in words).strip()
        else:
            text = r["text"]
        q = Quality()
        if dropped:
            q.notes.append(f"удалено дублей на границе: {len(dropped)}")
        if text and _HALLUCINATIONS.search(text) and len(text.split()) < 8:
            q.notes.append(f"подозрение на галлюцинацию ASR: «{text}»")
            text, words = "", []
        if not text:
            return [self.result(vad_start, end, [], M.insufficient_data, quality=q, latency_ms=lat, payload={"replace_key": f"seg{seg_id}", "segment_id": seg_id, "reason": "текст не распознан"})]
        if words:
            self.last_word_end = max(self.last_word_end, max(w.t_end_ms for w in words))
        seg_t0 = words[0].t_start_ms if words else vad_start
        seg_t1 = words[-1].t_end_ms if words else end
        self.history = (self.history + " " + text)[-400:]
        self.finals += 1
        metrics = [
            Metric(name="text", value=text, method="faster-whisper", kind="model_output"),
            Metric(name="word_count", value=len(text.split()), unit="words", method="ASR", kind="derived"),
        ]
        if r["avg_logprob"]:
            metrics.append(Metric(name="avg_logprob", value=round(float(np.mean(r["avg_logprob"])), 4), unit="log-prob", method="средний log-prob токенов Whisper (не калиброванная уверенность)", kind="model_output"))
            metrics.append(Metric(name="no_speech_prob", value=round(float(np.mean(r["no_speech_prob"])), 4), unit="prob", method="Whisper no_speech_prob", kind="model_output"))
        if words:
            metrics.append(Metric(name="mean_word_probability", value=round(float(np.mean([w.probability or 0 for w in words])), 4), unit="prob", method="средняя вероятность слов Whisper (не калиброванная)", kind="model_output"))
        sentence_end_ms = [w.t_end_ms for w in words if re.search(r"[.!?…]$", w.text)]
        payload: dict[str, Any] = {
            "replace_key": f"seg{seg_id}",
            "segment_id": seg_id,
            "text": text,
            "words": [{"w": w.text, "s": w.t_start_ms, "e": w.t_end_ms, "p": round(w.probability or 0, 3)} for w in words],
            "sentence_end_ms": sentence_end_ms,
            "vad_start_ms": int(vad_start),
            "vad_end_ms": int(end),
        }
        reference = None
        if self.tracker is not None:
            al = self.tracker.align(text)
            payload["alignment"] = al
            if al["status"] == "ok":
                N = al["ref_words"]
                metrics += [
                    Metric(name="wer_segment", value=al["wer"], unit="ratio", method="(S+D+I)/N на выровненном фрагменте", kind="derived", group="reading"),
                    Metric(name="substitutions", value=al["substitutions"], unit="words", method="выравнивание DP", kind="derived", group="reading"),
                    Metric(name="deletions", value=al["deletions"], unit="words", method="выравнивание DP", kind="derived", group="reading"),
                    Metric(name="insertions", value=al["insertions"], unit="words", method="выравнивание DP", kind="derived", group="reading"),
                    Metric(name="ref_words", value=N, unit="words", method="слов эталона во фрагменте", kind="derived", group="reading"),
                    Metric(name="reading_accuracy", value=round(1 - al["wer"], 4), unit="1-WER", normalized_score=round(100 * max(0.0, 1 - al["wer"]), 2), method="100·max(0, 1−WER) (ТЗ §6)", kind="derived", group="reading"),
                ]
                s_from, s_to = al["sentence_from"], al["sentence_to"]
                sentence = " ".join(self.tracker.sentence(i) for i in range(s_from, s_to + 1))
                reference = {"sentence_from": s_from, "sentence_to": s_to, "sentence": sentence, "progress": al["cumulative"]["progress"]}
                self.ctx.publish(TextReference(
                    type=EventType.text_reference, session_id=self.ctx.session_id, t_ms=int(seg_t0), t_end_ms=int(seg_t1),
                    sentence_index=s_to, sentence=sentence,
                    context_before=" ".join(self.tracker.sentence(i) for i in range(max(0, s_from - 2), s_from)),
                    context_after=self.tracker.sentence(s_to + 1), segment_id=seg_id,
                    sentence_from=s_from, sentences=[self.tracker.sentence(i) for i in range(s_from, s_to + 1)],
                ))
            else:
                metrics.append(Metric(name="wer_segment", value=None, unit="ratio", method="выравнивание DP", status=M.insufficient_data, kind="derived", group="reading"))
        self.ctx.publish(Transcript(
            type=EventType.transcript_final, session_id=self.ctx.session_id, t_ms=int(seg_t0), t_end_ms=int(seg_t1), text=text,
            words=words, segment_id=seg_id, language=self.language, reference=reference,
        ))
        return [self.result(seg_t0, seg_t1, metrics, M.ok, quality=q, latency_ms=lat, payload=payload)]

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        if self.tracker is None:
            return []
        cum = self.tracker.cumulative()
        return [self.result(0, 0, [
            Metric(name="wer_cumulative", value=cum["wer"], unit="ratio", method="Σ(S+D+I)/ΣN по выровненным фрагментам", kind="derived", group="reading",
                   status=M.ok if cum["wer"] is not None else M.insufficient_data),
            Metric(name="reading_progress", value=cum["progress"], unit="ratio", method="доля эталона, пройденная указателем", kind="derived", group="reading"),
        ], payload={"summary": True, "cumulative": cum})]
