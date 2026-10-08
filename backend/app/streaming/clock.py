"""Часы сессии: монотонное время относительно session_t0."""

from __future__ import annotations

import time


class SessionClock:
    def __init__(self) -> None:
        self.t0_ns: int | None = None

    def start(self) -> None:
        self.t0_ns = time.monotonic_ns()

    @property
    def started(self) -> bool:
        return self.t0_ns is not None

    def now_ms(self) -> float:
        if self.t0_ns is None:
            return 0.0
        return (time.monotonic_ns() - self.t0_ns) / 1e6

    def from_monotonic_ns(self, t_ns: int) -> float:
        assert self.t0_ns is not None
        return (t_ns - self.t0_ns) / 1e6
