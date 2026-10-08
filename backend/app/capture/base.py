"""Единый интерфейс источников медиаданных."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np


class SourceSink(Protocol):
    """Приёмник сырых данных (реализуется StreamProcessor). Вызывается из потоков источника."""

    def on_video(self, frame: np.ndarray, t_ms: float, pts_ms: float | None = None) -> None: ...

    def on_audio(
        self, samples: np.ndarray, sample_rate: int, t_ms: float, pts_ms: float | None = None
    ) -> None: ...

    def on_error(self, where: str, message: str, fatal: bool = False) -> None: ...

    def on_eof(self) -> None: ...


class MediaSource(ABC):
    kind: str = "base"

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.started = False
        self.info: dict[str, Any] = {}

    @abstractmethod
    def _run_threads(self, sink: SourceSink, clock: Any) -> list[threading.Thread]: ...

    def start(self, sink: SourceSink, clock: Any) -> None:
        if self.started:  # повторный Start не создаёт дублирующие потоки
            return
        self.started = True
        self._stop.clear()
        self._threads = self._run_threads(sink, clock)
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive() and t is not threading.current_thread():
                t.join(timeout)
        self._threads = []
        self.release()

    def release(self) -> None:
        """Освобождение устройств (переопределяется)."""

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)
