"""Изолированный прогон одного анализатора на событиях из файла (без сессии/UI)."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.analyzers.base import AnalyzerContext, AnalyzerHost, InferenceGate
from app.capture.file_replay import FileReplaySource
from app.config import load_config
from app.preprocessing.vad import VadStage
from app.schemas.events import EventType, ResultEvent
from app.schemas.results import AnalysisResult
from app.streaming.clock import SessionClock
from app.streaming.processor import StreamProcessor
from app.streaming.router import EventRouter


class Harness:
    def __init__(self, module_ids: list[str], mode: str = "reading", reference_text: str | None = None,
                 question: str | None = None, overrides: dict[str, Any] | None = None) -> None:
        # auto: gpu при наличии CUDA (целевая машина), иначе cpu; можно задать MBA_TEST_PROFILE
        self.cfg = load_config(os.environ.get("MBA_TEST_PROFILE", "auto"), overrides)
        self.module_ids = module_ids
        self.mode = mode
        self.reference_text = reference_text
        self.question = question
        self.router = EventRouter()
        self.results: list[AnalysisResult] = []
        self.events: list[Any] = []
        self.models: list[dict[str, Any]] = []

    async def run_file(self, path: Path, max_duration_s: float | None = None, with_vad: bool = True) -> list[AnalysisResult]:
        from app.analyzers.registry import load_class

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(4)
        ctx = AnalyzerContext(
            session_id="test", mode=self.mode, config=self.cfg, router=self.router, loop=loop, executor=executor,
            gate=InferenceGate(2), reference_text=self.reference_text, question=self.question,
            shared={"register_model": lambda **m: self.models.append(m)},
        )
        sub = self.router.subscribe("collect", list(EventType), maxsize=100000, policy="lossless")
        hosts = []
        for mid in self.module_ids:
            h = AnalyzerHost(load_class(mid)(ctx), self.cfg["modules"].get(mid, {}))
            await h.start()
            hosts.append(h)
        for h in hosts:
            await h.ready.wait()
        self.hosts = hosts
        vad = VadStage("test", self.router, self.cfg) if with_vad else None
        if vad:
            vad.start()
        clock = SessionClock()
        clock.start()
        done = asyncio.Event()
        proc = StreamProcessor("test", self.router, loop, clock, self.cfg, on_eof=done.set, backpressure=True)
        src = FileReplaySource(str(path), realtime=False, video_fps=self.cfg["capture"]["video_fps"], max_duration_s=max_duration_s)
        src.start(proc, clock)
        await done.wait()
        await asyncio.sleep(0.1)
        if vad:
            await vad.stop()
        for stage in (0, 1):
            from app.analyzers.registry import MODULES

            await asyncio.gather(*(h.stop(drain_timeout_s=600) for h in hosts if MODULES[h.plugin.module_id]["stage"] == stage))
        await asyncio.sleep(0.05)
        sub.close()
        async for ev in sub:
            self.events.append(ev)
            if isinstance(ev, ResultEvent):
                self.results.append(ev.result)
        executor.shutdown(wait=False)
        return self.results

    def by(self, module_id: str) -> list[AnalysisResult]:
        return [r for r in self.results if r.module_id == module_id]

    def of_type(self, t: EventType) -> list[Any]:
        return [e for e in self.events if e.type == t]
