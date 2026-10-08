"""Session / SessionManager: жизненный цикл сессии и оркестрация pipeline.

created → initializing → running → stopping → completed (или failed)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.analyzers.base import AnalyzerContext, AnalyzerHost, InferenceGate
from app.analyzers.llm_client import LLMUsage
from app.analyzers.registry import MODULES, PRODUCTION_MODULES, load_class
from app.capture.base import MediaSource
from app.config import data_dir, load_config
from app.fusion.crossmodal import session_alignment
from app.fusion.engine import TemporalFusionEngine
from app.preprocessing.vad import VadStage
from app.schemas.events import (
    EventType,
    ModuleErrorEvent,
    ResultEvent,
    SpeechActivity,
    TextReference,
    Transcript,
)
from app.schemas.results import AnalysisResult, ModuleHealth, ModuleState
from app.schemas.session import SessionCreate, SessionInfo, SessionState
from app.scoring.engine import RuleBasedScoring
from app.storage.db import Database, now_iso
from app.storage.media import MediaRecorder
from app.storage.results_store import ResultStore, write_parquet
from app.streaming.clock import SessionClock
from app.streaming.processor import StreamProcessor
from app.streaming.router import EventRouter

log = logging.getLogger(__name__)


class Session:
    def __init__(self, sid: str, req: SessionCreate, cfg: dict[str, Any], db: Database) -> None:
        self.id = sid
        self.req = req
        self.cfg = cfg
        self.db = db
        self.state = SessionState.created
        self.created_at = now_iso()
        self.started_at: str | None = None
        self.stopped_at: str | None = None
        self.error: str | None = None
        self.dir = data_dir(cfg) / "sessions" / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.router = EventRouter()
        self.clock = SessionClock()
        self.store = ResultStore()
        self.modules = self._resolve_modules()
        self.fusion = TemporalFusionEngine(sid, cfg, self.modules)
        self.llm_usage = LLMUsage(float(cfg["llm"].get("session_budget_usd") or 0))
        self.hosts: dict[str, AnalyzerHost] = {}
        self.unavailable: dict[str, ModuleHealth] = {}
        self.source: MediaSource | None = None
        self.processor: StreamProcessor | None = None
        self.vad: VadStage | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.gate: InferenceGate | None = None
        self.media_recorder: MediaRecorder | None = None
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.module_errors: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._startup_task: asyncio.Task[None] | None = None
        self._stop_lock = asyncio.Lock()
        self._stop_requested = False
        self._last_preview_sent: float | None = None
        self.end_media_ms: float = 0.0

    # --- конфигурация -----------------------------------------------------
    def _resolve_modules(self) -> list[str]:
        if self.req.modules is not None:
            mods = [m for m in self.req.modules if m in MODULES]
        else:
            mods = [m for m in PRODUCTION_MODULES if self.cfg["modules"].get(m, {}).get("enabled", True)]
        return mods

    def info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            state=self.state,
            mode=self.req.mode,
            source=self.req.source,
            created_at=self.created_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            error=self.error,
            config={"modules": self.modules, "profile": self.cfg.get("profile_resolved")},
            stats=self.summary.get("capture", {}),
        )

    def _set_state(self, st: SessionState, **db_fields: Any) -> None:
        self.state = st
        self.db.update_session(self.id, state=st.value, **db_fields)
        self.broadcast({"type": "session", "data": self.info().model_dump(mode="json")})

    # --- WebSocket fan-out ------------------------------------------------
    def add_client(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self.clients.add(q)
        return q

    def remove_client(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self.clients.discard(q)

    def broadcast(self, msg: dict[str, Any]) -> None:
        for q in list(self.clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    # --- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        if self.state != SessionState.created:
            return  # повторный Start не создаёт дублирующих потоков
        self._set_state(SessionState.initializing)
        self._startup_task = asyncio.create_task(self._startup(), name=f"startup:{self.id}")

    async def _startup(self) -> None:
        loop = asyncio.get_running_loop()
        cfg = self.cfg
        try:
            self.executor = ThreadPoolExecutor(max_workers=int(cfg["inference"]["threads"]), thread_name_prefix=f"inf-{self.id}")
            self.gate = InferenceGate(int(cfg["inference"]["max_parallel"]))
            ctx = AnalyzerContext(
                session_id=self.id,
                mode=self.req.mode,
                config=cfg,
                router=self.router,
                loop=loop,
                executor=self.executor,
                gate=self.gate,
                reference_text=self.req.reference_text,
                question=self.req.question,
                shared={"register_model": self.db.register_model, "llm_usage": self.llm_usage},
            )
            self._tasks.append(asyncio.create_task(self._collector(), name="collector"))
            for mid in self.modules:
                try:
                    cls = load_class(mid)
                except Exception as e:  # noqa: BLE001
                    log.exception("import failed: %s", mid)
                    self.unavailable[mid] = ModuleHealth(module_id=mid, state=ModuleState.unavailable, detail=f"import: {type(e).__name__}: {e}")
                    continue
                host = AnalyzerHost(cls(ctx), cfg["modules"].get(mid, {}))
                self.hosts[mid] = host
                await host.start()
            self.broadcast({"type": "status", "data": self.status()})
            # ждём загрузки моделей (первый запуск может скачивать веса)
            while not all(h.ready.is_set() for h in self.hosts.values()) and not self._stop_requested:
                await asyncio.sleep(0.5)
                self.broadcast({"type": "status", "data": self.status()})
            if self._stop_requested:
                return
            # источник
            self.source = self._make_source()
            if self.req.save_media and self.req.source == "live":
                self.media_recorder = MediaRecorder(self.dir / "media", video_fps=int(cfg["modules"]["face"]["analysis_fps"]))
            self.vad = VadStage(self.id, self.router, cfg)
            self.vad.start()
            self.clock.start()
            self.processor = StreamProcessor(
                self.id, self.router, loop, self.clock, cfg, on_eof=self._on_eof, media_recorder=self.media_recorder,
                backpressure=self.req.source == "file" and self.req.file is not None and not self.req.file.realtime,
            )
            self.source.start(self.processor, self.clock)
            self.started_at = now_iso()
            self._set_state(SessionState.running, started_at=self.started_at)
            self._tasks.append(asyncio.create_task(self._ticker(), name="ticker"))
        except Exception as e:  # noqa: BLE001
            log.exception("startup failed")
            self.error = f"{type(e).__name__}: {e}"
            await self._teardown()
            self._set_state(SessionState.failed, error=self.error)

    def _make_source(self) -> MediaSource:
        cap = self.cfg["capture"]
        if self.req.source == "file":
            from app.capture.file_replay import FileReplaySource

            if not self.req.file:
                raise ValueError("не указан файл для File Replay")
            return FileReplaySource(
                self.req.file.path,
                realtime=self.req.file.realtime,
                speed=self.req.file.speed,
                video_fps=float(cap["video_fps"]),
            )
        from app.capture.windows_live import WindowsLiveSource

        live = self.req.live
        region = None
        if live and live.region:
            r = live.region
            region = (r.left, r.top, r.left + r.width, r.top + r.height)
        return WindowsLiveSource(
            monitor=live.monitor if live else 0,
            region=region,
            audio_device=live.audio_device if live else None,
            fps=live.fps if live else int(cap["video_fps"]),
            audio_chunk_ms=int(cap["audio_chunk_ms"]),
        )

    def _on_eof(self) -> None:
        if self.state == SessionState.running:
            asyncio.create_task(self.stop(reason="eof"))

    async def stop(self, reason: str = "user") -> None:
        async with self._stop_lock:
            if self.state in (SessionState.completed, SessionState.failed, SessionState.stopping):
                return
            if self.state == SessionState.created:
                self._set_state(SessionState.completed, stopped_at=now_iso())
                return
            self._stop_requested = True
            if self._startup_task and not self._startup_task.done():
                await self._startup_task
            if self.state == SessionState.failed:
                return
            self._set_state(SessionState.stopping)
            try:
                await self._shutdown_pipeline()
                self.stopped_at = now_iso()
                self._set_state(SessionState.completed, stopped_at=self.stopped_at, stats=self.summary)
            except Exception as e:  # noqa: BLE001
                log.exception("stop failed")
                self.error = f"{type(e).__name__}: {e}"
                self._set_state(SessionState.failed, error=self.error, stats=self.summary)
            log.info("session %s stopped (%s)", self.id, reason)

    def _final_capture_status(self) -> dict[str, Any]:
        if self.processor is None:
            return {}
        final = self.processor.status()  # итоговые счётчики (после flush)
        snap = getattr(self, "_capture_snapshot", {}) or {}
        for k in ("video", "audio"):
            if k in snap:
                final[k]["no_signal"] = snap[k]["no_signal"]
                final[k]["fps"] = snap[k].get("fps", final[k].get("fps"))
        return final

    async def _shutdown_pipeline(self) -> None:
        loop = asyncio.get_running_loop()
        # снимок состояния захвата до остановки (после Stop «нет сигнала» было бы артефактом)
        self._capture_snapshot = self.processor.status() if self.processor else {}
        # 1. источник и устройства
        if self.source is not None:
            await loop.run_in_executor(None, self.source.stop)
        if self.processor is not None:
            self.processor.finish()
            self.end_media_ms = max(self.clock.now_ms() if self.req.source == "live" else 0.0, self._media_end())
        await asyncio.sleep(0.05)  # дать call_soon_threadsafe доставить последние события
        if self.media_recorder is not None:
            files = self.media_recorder.close()
            self.summary["media_files"] = files
        # 2. VAD, затем анализаторы по стадиям (производители → потребители)
        if self.vad is not None:
            await self.vad.stop()
        for stage in sorted({MODULES[m]["stage"] for m in self.hosts}):
            await asyncio.gather(*(h.stop() for m, h in self.hosts.items() if MODULES[m]["stage"] == stage))
        # 3. collector
        await asyncio.sleep(0.05)
        self.router.close_all()
        for t in self._tasks:
            try:
                await asyncio.wait_for(t, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                t.cancel()
        # 4. fusion → cross-modal → scoring → storage
        for r in self.fusion.finalize(self.end_media_ms):
            self.store.add(r)
        self._finalize_analysis()
        await loop.run_in_executor(None, self._persist)
        await self._teardown()

    def _media_end(self) -> float:
        p = self.processor
        ends = [0.0]
        if p is not None:
            if p.last_video_t_ms is not None:
                ends.append(p.last_video_t_ms)
            ends.append(p.audio_samples_out / 16.0)
        return max(ends)

    async def _teardown(self) -> None:
        if self.source is not None and self.source.started:
            await asyncio.get_running_loop().run_in_executor(None, self.source.stop)
        for h in self.hosts.values():
            if h.task and not h.task.done():
                h.task.cancel()
        for t in self._tasks:
            if not t.done():
                t.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    # --- фоновые задачи ---------------------------------------------------
    async def _collector(self) -> None:
        sub = self.router.subscribe(
            "collector",
            [
                EventType.analysis_result,
                EventType.module_error,
                EventType.transcript_partial,
                EventType.transcript_final,
                EventType.speech_activity,
                EventType.text_reference,
            ],
            maxsize=int(self.cfg["queues"]["results"]),
            policy="lossless",
        )
        async for ev in sub:
            if isinstance(ev, ResultEvent):
                r = ev.result
                self.store.add(r)
                self.fusion.add(r)
                self.broadcast({"type": "result", "data": _light(r)})
            elif isinstance(ev, ModuleErrorEvent):
                e = {"module_id": ev.module_id, "message": ev.message, "fatal": ev.fatal, "t_ms": ev.t_ms}
                self.module_errors.append(e)
                self.broadcast({"type": "module_error", "data": e})
            elif isinstance(ev, Transcript):
                self.broadcast(
                    {
                        "type": ev.type.value,
                        "data": {
                            "t_start_ms": ev.t_ms,
                            "t_end_ms": ev.t_end_ms,
                            "text": ev.text,
                            "segment_id": ev.segment_id,
                            "reference": ev.reference,
                        },
                    }
                )
            elif isinstance(ev, SpeechActivity):
                self.broadcast({"type": "speech.activity", "data": {"kind": ev.kind, "t_start_ms": ev.t_ms, "t_end_ms": ev.t_end_ms, "segment_id": ev.segment_id}})
            elif isinstance(ev, TextReference):
                self.broadcast(
                    {"type": "text.reference", "data": {"t_start_ms": ev.t_ms, "t_end_ms": ev.t_end_ms, "sentence_index": ev.sentence_index, "sentence": ev.sentence}}
                )

    async def _ticker(self) -> None:
        step = self.cfg["fusion"]["step_ms"] / 1000
        status_period = self.cfg["capture"]["status_interval_ms"] / 1000
        last_status = 0.0
        while self.state == SessionState.running:
            now_media = self._media_now()
            for r in self.fusion.tick(now_media):
                self.store.add(r)
                self.broadcast({"type": "result", "data": _light(r)})
            if time.monotonic() - last_status >= status_period:
                last_status = time.monotonic()
                self.broadcast({"type": "status", "data": self.status()})
            p = self.processor
            if p is not None and p.latest_preview is not None and p.latest_preview_t_ms != self._last_preview_sent:
                self._last_preview_sent = p.latest_preview_t_ms
                self.broadcast(
                    {"type": "preview", "data": {"t_ms": p.latest_preview_t_ms, "jpeg_b64": base64.b64encode(p.latest_preview).decode()}}
                )
            await asyncio.sleep(min(step, 0.5))

    def _media_now(self) -> float:
        """Текущее медиавремя: по реально доставленным данным (для файла может отличаться от часов)."""
        if self.req.source == "live":
            return self.clock.now_ms()
        return self._media_end()

    # --- итоговый анализ --------------------------------------------------
    def _finalize_analysis(self) -> None:
        results = self.store.results
        windows: list[dict[str, Any]] = []
        for r in results:
            if r.module_id == "fusion":
                windows.append({m.name: m.value for m in r.metrics})
        pauses: list[tuple[int, int]] = []
        boundaries: list[int] = []
        for r in results:
            if r.module_id == "prosody" and r.payload.get("pauses"):
                pauses.extend((int(a), int(b)) for a, b in r.payload["pauses"])
            if r.module_id == "speech" and r.finality == "final":
                boundaries.extend(int(b) for b in r.payload.get("sentence_end_ms", []))
        pauses = sorted(set(pauses))
        crossmodal = session_alignment(windows, pauses, sorted(set(boundaries)))
        scores = RuleBasedScoring().score(results, crossmodal, {**self.cfg, "composite_enabled": self.req.composite_enabled})
        self.summary.update(
            {
                "capture": self._final_capture_status(),
                "vad": self.vad.status() if self.vad else {},
                "router": self.router.stats(),
                "modules": [h.model_dump(mode="json") for h in self.module_health()],
                "module_errors": self.module_errors[-200:],
                "fusion": {"windows": len(self.fusion.computed), "recomputed": self.fusion.recomputed},
                "llm_usage": {**self.llm_usage.summary(), "backend": self.cfg["llm"].get("backend")},
                "crossmodal": crossmodal,
                "scores": scores,
                "media_end_ms": round(self.end_media_ms),
                "config": self.cfg,
                "request": self.req.model_dump(mode="json"),
            }
        )

    def _persist(self) -> None:
        results = self.store.dicts()
        self.db.replace_results(self.id, results)
        if self.cfg["storage"].get("parquet", True):
            self.summary["parquet"] = write_parquet(results, self.store.raw_frames, self.dir)
        with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=1, default=str)

    # --- статус -----------------------------------------------------------
    def module_health(self) -> list[ModuleHealth]:
        out = [h.plugin.health() for h in self.hosts.values()]
        out += list(self.unavailable.values())
        all_mods = set(PRODUCTION_MODULES)
        for m in sorted(all_mods - set(self.modules)):
            out.append(ModuleHealth(module_id=m, state=ModuleState.disabled))
        return out

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "state": self.state.value,
            "t_ms": round(self.clock.now_ms()),
            "media_ms": round(self._media_now()) if self.processor else 0,
            "capture": self.processor.status() if self.processor else None,
            "vad": self.vad.status() if self.vad else None,
            "modules": [h.model_dump(mode="json") for h in self.module_health()],
            "gate": {"slots": self.gate.slots, "active": self.gate.active, "waiting": self.gate.waiting} if self.gate else None,
            "router": self.router.stats(),
            "results": len(self.store.results),
            "llm_usage": {**self.llm_usage.summary(), "backend": self.cfg["llm"].get("backend")},
        }


def _light(r: AnalysisResult) -> dict[str, Any]:
    d = r.model_dump(mode="json")
    d.get("payload", {}).pop("raw", None)
    d.get("payload", {}).pop("embedding", None)
    return d


class SessionManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.sessions: dict[str, Session] = {}

    def create(self, req: SessionCreate) -> Session:
        cfg = load_config(req.profile, req.overrides)
        sid = uuid.uuid4().hex[:12]
        s = Session(sid, req, cfg, self.db)
        self.db.insert_session(
            {
                "id": sid,
                "state": s.state.value,
                "mode": req.mode,
                "source": req.source,
                "created_at": s.created_at,
                "request": req.model_dump(mode="json"),
                "config": cfg,
            }
        )
        self.sessions[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    def active(self) -> list[Session]:
        return [s for s in self.sessions.values() if s.state in (SessionState.initializing, SessionState.running, SessionState.stopping)]

    async def shutdown(self) -> None:
        for s in list(self.sessions.values()):
            await s.stop(reason="shutdown")


def load_session_results(db: Database, sid: str) -> dict[str, Any] | None:
    row = db.get_session(sid)
    if row is None:
        return None
    return {"session": row, "results": db.load_results(sid), "summary": row.get("stats", {})}


def session_dir(cfg: dict[str, Any], sid: str) -> Path:
    return data_dir(cfg) / "sessions" / sid
