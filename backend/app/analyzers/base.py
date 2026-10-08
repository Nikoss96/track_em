"""Контракт AnalyzerPlugin и хост, исполняющий плагин в собственном worker-цикле."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from app.schemas.events import Event, EventType, ModuleErrorEvent, ResultEvent
from app.schemas.results import AnalysisResult, ModuleHealth, ModuleState, ResultStatus
from app.streaming.router import EventRouter, Policy, Subscription

log = logging.getLogger(__name__)
T = TypeVar("T")


class ModelUnavailable(RuntimeError):
    """Модель/бэкенд недоступны в окружении — модуль переходит в `unavailable`."""


class InferenceGate:
    """Ограничивает число одновременных тяжёлых inference-задач (CPU/VRAM)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self._sem = asyncio.Semaphore(slots)
        self.waiting = 0
        self.active = 0

    async def __aenter__(self) -> InferenceGate:
        self.waiting += 1
        await self._sem.acquire()
        self.waiting -= 1
        self.active += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.active -= 1
        self._sem.release()


@dataclass
class AnalyzerContext:
    session_id: str
    mode: str  # reading | qa
    config: dict[str, Any]  # полный конфиг сессии
    router: EventRouter
    loop: asyncio.AbstractEventLoop
    executor: ThreadPoolExecutor
    gate: InferenceGate
    reference_text: str | None = None
    question: str | None = None
    shared: dict[str, Any] = field(default_factory=dict)

    def publish(self, ev: Event) -> None:
        """Потокобезопасная публикация события."""
        if _in_loop(self.loop):
            self.router.publish(ev)
        else:
            self.loop.call_soon_threadsafe(self.router.publish, ev)


def _in_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


class AnalyzerPlugin(ABC):
    module_id: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    subscribes: ClassVar[tuple[EventType, ...]] = ()
    queue_policy: ClassVar[Policy] = "drop_oldest"
    queue_size: ClassVar[int] = 32

    def __init__(self, ctx: AnalyzerContext) -> None:
        self.ctx = ctx
        self.cfg: dict[str, Any] = {}
        self.model_id = "none"
        self.model_version = "none"
        self.state = ModuleState.loading
        self.detail: str | None = None
        self.processed = 0
        self.errors = 0
        self.last_latency_ms: float | None = None
        self._lat_sum = 0.0
        self.sub: Subscription | None = None

    # --- контракт ---------------------------------------------------------
    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    async def process(self, event: Event) -> list[AnalysisResult]: ...

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        return []

    def health(self) -> ModuleHealth:
        return ModuleHealth(
            module_id=self.module_id,
            state=self.state,
            model_id=self.model_id,
            model_version=self.model_version,
            detail=self.detail,
            processed=self.processed,
            dropped=self.sub.dropped if self.sub else 0,
            errors=self.errors,
            queue_size=self.sub.qsize() if self.sub else 0,
            last_latency_ms=self.last_latency_ms,
            avg_latency_ms=(self._lat_sum / self.processed) if self.processed else None,
        )

    async def shutdown(self) -> None:
        return None

    # --- помощники --------------------------------------------------------
    async def run_blocking(self, fn: Callable[..., T], *args: Any, gated: bool = True) -> T:
        if gated:
            async with self.ctx.gate:
                return await self.ctx.loop.run_in_executor(self.ctx.executor, fn, *args)
        return await self.ctx.loop.run_in_executor(self.ctx.executor, fn, *args)

    def result(
        self,
        t_start_ms: int,
        t_end_ms: int,
        metrics: list,
        status: ResultStatus = ResultStatus.ok,
        **kw: Any,
    ) -> AnalysisResult:
        return AnalysisResult(
            session_id=self.ctx.session_id,
            module_id=self.module_id,
            model_id=self.model_id,
            model_version=self.model_version,
            t_start_ms=int(t_start_ms),
            t_end_ms=int(max(t_end_ms, t_start_ms)),
            metrics=metrics,
            status=status,
            **kw,
        )

    def _record_latency(self, ms: float) -> None:
        self.processed += 1
        self.last_latency_ms = ms
        self._lat_sum += ms


class AnalyzerHost:
    """Независимый worker для одного плагина. Ошибка плагина не останавливает остальных."""

    MAX_CONSECUTIVE_ERRORS = 20

    def __init__(self, plugin: AnalyzerPlugin, module_cfg: dict[str, Any]) -> None:
        self.plugin = plugin
        self.module_cfg = module_cfg
        self.task: asyncio.Task[None] | None = None
        self.ready = asyncio.Event()

    async def start(self) -> None:
        p = self.plugin
        ctx = p.ctx
        p.sub = ctx.router.subscribe(
            p.module_id,
            p.subscribes,
            maxsize=p.queue_size,
            policy=p.queue_policy,
            key=getattr(p, "dedupe_key", None),
        )
        self.task = asyncio.create_task(self._run(), name=f"analyzer:{p.module_id}")

    async def _init(self) -> bool:
        p = self.plugin
        p.state = ModuleState.loading
        try:
            await p.initialize(self.module_cfg)
            p.state = ModuleState.running
            return True
        except ModelUnavailable as e:
            p.state = ModuleState.unavailable
            p.detail = str(e)
        except Exception as e:  # noqa: BLE001
            log.exception("init failed: %s", p.module_id)
            p.state = ModuleState.error
            p.detail = f"{type(e).__name__}: {e}"
        p.ctx.publish(
            ModuleErrorEvent(
                type=EventType.module_error,
                session_id=p.ctx.session_id,
                t_ms=0,
                module_id=p.module_id,
                message=p.detail or "init failed",
                fatal=True,
            )
        )
        return False

    async def _run(self) -> None:
        p = self.plugin
        ok = await self._init()
        self.ready.set()
        if not ok:
            # Модуль недоступен: снимаем подписку, чтобы не копить очередь.
            if p.sub:
                p.ctx.router.unsubscribe(p.sub)
            return
        consecutive = 0
        assert p.sub is not None
        async for ev in p.sub:
            t0 = time.perf_counter()
            try:
                results = await p.process(ev)
                consecutive = 0
            except Exception as e:  # noqa: BLE001
                consecutive += 1
                p.errors += 1
                p.detail = f"{type(e).__name__}: {e}"
                log.exception("process failed: %s", p.module_id)
                p.ctx.publish(
                    ModuleErrorEvent(
                        type=EventType.module_error,
                        session_id=p.ctx.session_id,
                        t_ms=ev.t_ms,
                        module_id=p.module_id,
                        message=p.detail,
                    )
                )
                if consecutive >= self.MAX_CONSECUTIVE_ERRORS:
                    p.state = ModuleState.error
                continue
            lat = (time.perf_counter() - t0) * 1000
            for r in results or []:
                if not r.latency_ms:
                    r.latency_ms = round(lat, 1)
                self._emit(r)
            if results:
                p._record_latency(lat)

    def _emit(self, r: AnalysisResult) -> None:
        self.plugin.ctx.publish(
            ResultEvent(type=EventType.analysis_result, session_id=r.session_id, t_ms=r.t_start_ms, result=r)
        )

    async def stop(self, drain_timeout_s: float = 30.0) -> None:
        """Закрывает подписку, ждёт обработки остатка очереди, вызывает finalize."""
        p = self.plugin
        await self.ready.wait()
        if p.sub and not p.sub.closed:
            p.sub.close()
        if self.task:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=drain_timeout_s)
            except TimeoutError:
                log.warning("drain timeout: %s", p.module_id)
                self.task.cancel()
        if p.state in (ModuleState.running, ModuleState.idle):
            try:
                for r in await p.finalize(p.ctx.session_id):
                    self._emit(r)
            except Exception:  # noqa: BLE001
                log.exception("finalize failed: %s", p.module_id)
        try:
            await p.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("shutdown failed: %s", p.module_id)
        if p.state == ModuleState.running:
            p.state = ModuleState.idle
