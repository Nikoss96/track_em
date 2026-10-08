"""Сериализуемые контракты результатов анализаторов (Pydantic v2)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResultStatus(str, Enum):
    ok = "ok"
    partial = "partial"
    insufficient_data = "insufficient_data"
    unavailable = "unavailable"
    not_applicable = "not_applicable"
    error = "error"


class Metric(BaseModel):
    """Одна метрика. value — сырое значение; normalized_score — только по явной процедуре."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | str | bool | list[Any] | dict[str, Any] | None
    unit: str | None = None
    normalized_score: float | None = Field(default=None, ge=0, le=100)
    # Только если модель действительно даёт интерпретируемую уверенность.
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str
    status: ResultStatus = ResultStatus.ok
    # Группа: measured (физическое измерение) | model_output | derived | experimental
    kind: str = "measured"
    group: str | None = None


class Quality(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_ok: bool = True
    notes: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    module_id: str
    model_id: str
    model_version: str
    t_start_ms: int
    t_end_ms: int
    metrics: list[Metric] = Field(default_factory=list)
    status: ResultStatus = ResultStatus.ok
    quality: Quality = Field(default_factory=Quality)
    latency_ms: float = 0.0
    error: str | None = None
    # provisional | final — для ASR/NLP/fusion
    finality: str = "final"
    # Дополнительные структурированные данные (текст, выравнивание, объяснения).
    payload: dict[str, Any] = Field(default_factory=dict)

    def metric(self, name: str) -> Metric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None


class ModuleState(str, Enum):
    disabled = "disabled"
    loading = "loading"
    running = "running"
    idle = "idle"
    unavailable = "unavailable"
    error = "error"


class ModuleHealth(BaseModel):
    module_id: str
    state: ModuleState
    model_id: str | None = None
    model_version: str | None = None
    detail: str | None = None
    processed: int = 0
    dropped: int = 0
    errors: int = 0
    queue_size: int = 0
    last_latency_ms: float | None = None
    avg_latency_ms: float | None = None
