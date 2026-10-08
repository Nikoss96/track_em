"""Внутренние события pipeline. Большие массивы — NumPy, не JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from app.schemas.results import AnalysisResult


class EventType(str, Enum):
    video_frame = "video.frame"
    audio_chunk = "audio.chunk"
    speech_activity = "speech.activity"
    transcript_partial = "transcript.partial"
    transcript_final = "transcript.final"
    text_reference = "text.reference"
    analysis_result = "analysis.result"
    session_status = "session.status"
    module_error = "module.error"


@dataclass(slots=True)
class Event:
    type: EventType
    session_id: str
    t_ms: int  # начало события относительно session_t0


@dataclass(slots=True)
class VideoFrame(Event):
    frame: np.ndarray  # HxWx3 uint8 RGB
    seq: int
    source_pts_ms: float | None = None
    corrupted: bool = False


@dataclass(slots=True)
class AudioChunk(Event):
    """Mono float32 [-1,1], 16 kHz (pcm16k) + метаданные исходного потока."""

    pcm16k: np.ndarray
    duration_ms: float
    seq: int
    source_rate: int
    source_channels: int
    rms_dbfs: float
    gap_before_ms: float = 0.0  # явная маркировка потерянного аудио перед чанком
    source_pts_ms: float | None = None

    @property
    def t_end_ms(self) -> int:
        return int(round(self.t_ms + self.duration_ms))


@dataclass(slots=True)
class SpeechActivity(Event):
    """kind=start: начало речевого сегмента; kind=end: [t_ms, t_end_ms] — завершённый сегмент."""

    kind: str  # start | end
    t_end_ms: int
    segment_id: int
    forced: bool = False  # сегмент разрезан по max_segment_ms (речь продолжается)
    mean_probability: float | None = None


@dataclass(slots=True)
class Word:
    text: str
    t_start_ms: int
    t_end_ms: int
    probability: float | None = None


@dataclass(slots=True)
class Transcript(Event):
    """transcript.partial / transcript.final; t_ms — начало фразы."""

    t_end_ms: int
    text: str
    words: list[Word] = field(default_factory=list)
    segment_id: int = 0
    language: str | None = None
    # Привязка к эталону (Reading Mode), если есть.
    reference: dict[str, Any] | None = None


@dataclass(slots=True)
class TextReference(Event):
    """Фрагмент эталонного текста, соответствующий прочитанному отрезку."""

    t_end_ms: int
    sentence_index: int
    sentence: str
    context_before: str
    context_after: str
    segment_id: int = 0
    sentence_from: int = -1
    sentences: list[str] = field(default_factory=list)  # предложения эталона [sentence_from..sentence_index]


@dataclass(slots=True)
class ResultEvent(Event):
    result: AnalysisResult


@dataclass(slots=True)
class SessionStatusEvent(Event):
    state: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModuleErrorEvent(Event):
    module_id: str
    message: str
    fatal: bool = False
