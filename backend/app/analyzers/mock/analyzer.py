"""MockAnalyzer — ТОЛЬКО для тестов инфраструктуры. Не входит в production-набор модулей.

Возвращает не «эмоции», а тривиальные проверяемые величины о входе
(число кадров/чанков, RMS), чтобы тесты проверяли прохождение pipeline.
"""

from __future__ import annotations

from typing import Any

from app.analyzers.base import AnalyzerPlugin
from app.schemas.events import AudioChunk, Event, EventType, VideoFrame
from app.schemas.results import AnalysisResult, Metric


class MockAnalyzer(AnalyzerPlugin):
    module_id = "mock"
    subscribes = (EventType.video_frame, EventType.audio_chunk)
    queue_policy = "lossless"
    queue_size = 1000

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.model_id = "mock"
        self.model_version = "test-only"
        self.fail_on = config.get("fail_on_seq")
        self.frames = 0
        self.chunks = 0

    async def process(self, event: Event) -> list[AnalysisResult]:
        if isinstance(event, VideoFrame):
            self.frames += 1
            if self.fail_on is not None and event.seq == self.fail_on:
                raise RuntimeError("mock failure")
            return [
                self.result(
                    event.t_ms,
                    event.t_ms,
                    [Metric(name="frame_seq", value=event.seq, method="mock", kind="derived")],
                )
            ]
        if isinstance(event, AudioChunk):
            self.chunks += 1
            return [
                self.result(
                    event.t_ms,
                    event.t_end_ms,
                    [Metric(name="rms_dbfs", value=event.rms_dbfs, unit="dBFS", method="mock")],
                )
            ]
        return []

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        return [
            self.result(
                0,
                0,
                [
                    Metric(name="frames", value=self.frames, method="mock", kind="derived"),
                    Metric(name="chunks", value=self.chunks, method="mock", kind="derived"),
                ],
            )
        ]
