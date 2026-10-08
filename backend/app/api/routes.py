"""REST API и WebSocket."""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from app.analyzers.registry import MODULES, PRODUCTION_MODULES
from app.capture.devices import list_audio_outputs, list_monitors
from app.config import cuda_available, data_dir, gpu_memory, load_config, read_secrets, save_secrets, save_user_config
from app.schemas.session import SessionCreate, SessionState
from app.session.manager import SessionManager, load_session_results
from app.storage.results_store import to_csv

api = APIRouter(prefix="/api")
ws_router = APIRouter()


def _mgr(request: Request) -> SessionManager:
    return request.app.state.manager


@api.get("/health")
async def health(request: Request) -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "status": "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": cuda_available(),
        "gpus": await asyncio.get_running_loop().run_in_executor(None, gpu_memory),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_used_gb": round(vm.used / 2**30, 1),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        "active_sessions": [s.id for s in _mgr(request).active()],
    }


@api.get("/devices")
async def devices() -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    monitors, audio = await asyncio.gather(loop.run_in_executor(None, list_monitors), loop.run_in_executor(None, list_audio_outputs))
    return {"monitors": monitors, "audio_outputs": audio}


@api.get("/modules")
async def modules(request: Request) -> dict[str, Any]:
    cfg = load_config()
    out = []
    for mid in PRODUCTION_MODULES:
        out.append({"id": mid, "title": MODULES[mid]["title"], "enabled": cfg["modules"].get(mid, {}).get("enabled", True), "config": cfg["modules"].get(mid, {})})
    return {"modules": out, "models": request.app.state.db.list_models(), "profile": cfg["profile_resolved"]}


@api.get("/config")
async def get_config() -> dict[str, Any]:
    return load_config()


@api.put("/config")
async def put_config(patch: dict[str, Any]) -> dict[str, Any]:
    save_user_config(patch)
    return load_config()


@api.get("/secrets")
async def get_secrets() -> dict[str, Any]:
    """Только факт наличия ключа; сам ключ не возвращается."""
    import os

    s = read_secrets()
    return {"openai_api_key_set": bool(s.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")),
            "openai_api_key_source": "env" if os.environ.get("OPENAI_API_KEY") else ("file" if s.get("openai_api_key") else None)}


@api.put("/secrets")
async def put_secrets(patch: dict[str, Any]) -> dict[str, Any]:
    allowed = {"openai_api_key"}
    bad = set(patch) - allowed
    if bad:
        raise HTTPException(400, f"неизвестные ключи: {sorted(bad)}")
    save_secrets(patch)
    return await get_secrets()


@api.post("/files")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    up = data_dir(load_config()) / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.bin").name
    dest = up / f"{uuid.uuid4().hex[:8]}_{name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    info: dict[str, Any] = {"path": str(dest), "size": dest.stat().st_size}
    if dest.suffix.lower() in (".mp4", ".mkv", ".wav", ".mov", ".webm", ".mp3", ".flac"):
        from app.capture.file_replay import probe

        try:
            info["probe"] = probe(str(dest))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"не удалось прочитать медиафайл: {e}") from e
    return info


@api.get("/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    return request.app.state.db.list_sessions()


@api.post("/sessions")
async def create_session(req: SessionCreate, request: Request) -> dict[str, Any]:
    if req.source == "file" and (not req.file or not Path(req.file.path).exists()):
        raise HTTPException(400, "файл для File Replay не найден")
    s = _mgr(request).create(req)
    return s.info().model_dump(mode="json")


@api.get("/sessions/{sid}")
async def get_session(sid: str, request: Request) -> dict[str, Any]:
    s = _mgr(request).get(sid)
    if s is not None:
        return {**s.info().model_dump(mode="json"), "status": s.status()}
    row = request.app.state.db.get_session(sid)
    if row is None:
        raise HTTPException(404, "сессия не найдена")
    return row


@api.post("/sessions/{sid}/start")
async def start_session(sid: str, request: Request) -> dict[str, Any]:
    s = _mgr(request).get(sid)
    if s is None:
        raise HTTPException(404, "сессия не найдена")
    others = [a for a in _mgr(request).active() if a.id != sid]
    if others and s.req.source == "live" and any(o.req.source == "live" for o in others):
        raise HTTPException(409, "уже идёт другая Live-сессия")
    await s.start()
    return s.info().model_dump(mode="json")


@api.post("/sessions/{sid}/stop")
async def stop_session(sid: str, request: Request) -> dict[str, Any]:
    s = _mgr(request).get(sid)
    if s is None:
        raise HTTPException(404, "сессия не найдена")
    asyncio.create_task(s.stop())
    await asyncio.sleep(0)
    return s.info().model_dump(mode="json")


@api.get("/sessions/{sid}/results")
async def results(sid: str, request: Request, t_from: int | None = None, t_to: int | None = None, module: str | None = None) -> dict[str, Any]:
    s = _mgr(request).get(sid)
    if s is not None and s.state not in (SessionState.completed, SessionState.failed):
        data = {"session": s.info().model_dump(mode="json"), "results": s.store.dicts(), "summary": {}}
    else:
        data = load_session_results(request.app.state.db, sid)
        if data is None:
            raise HTTPException(404, "сессия не найдена")
    rs = data["results"]
    if module:
        rs = [r for r in rs if r["module_id"] == module]
    if t_from is not None:
        rs = [r for r in rs if r["t_end_ms"] >= t_from]
    if t_to is not None:
        rs = [r for r in rs if r["t_start_ms"] <= t_to]
    data["results"] = rs
    return data


@api.get("/sessions/{sid}/export")
async def export(sid: str, request: Request, format: str = Query("json", pattern="^(json|csv|parquet)$")) -> Response:
    data = load_session_results(request.app.state.db, sid)
    if data is None:
        raise HTTPException(404, "сессия не найдена")
    if data["session"]["state"] not in ("completed", "failed"):
        raise HTTPException(409, "экспорт доступен после завершения сессии")
    if format == "csv":
        return Response(
            to_csv(data["results"]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="session_{sid}_metrics.csv"'},
        )
    if format == "parquet":
        p = data_dir(load_config()) / "sessions" / sid / "metrics.parquet"
        if not p.exists():
            raise HTTPException(404, "parquet не найден")
        return FileResponse(p, filename=f"session_{sid}_metrics.parquet")
    body = json.dumps(data, ensure_ascii=False, default=str)
    return Response(
        body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="session_{sid}.json"'},
    )


@api.get("/sessions/{sid}/preview")
async def preview(sid: str, request: Request) -> Response:
    s = _mgr(request).get(sid)
    if s is None or s.processor is None or s.processor.latest_preview is None:
        raise HTTPException(404, "превью нет")
    return Response(s.processor.latest_preview, media_type="image/jpeg")


@ws_router.websocket("/ws/sessions/{sid}")
async def ws_session(websocket: WebSocket, sid: str) -> None:
    mgr: SessionManager = websocket.app.state.manager
    s = mgr.get(sid)
    await websocket.accept()
    if s is None:
        await websocket.send_json({"type": "error", "data": {"message": "сессия не активна"}})
        await websocket.close()
        return
    q = s.add_client()
    try:
        await websocket.send_json({"type": "session", "data": s.info().model_dump(mode="json")})
        await websocket.send_json({"type": "status", "data": s.status()})
        while True:
            msg = await q.get()
            await websocket.send_text(json.dumps(msg, ensure_ascii=False, default=str))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        s.remove_client(q)


__all__ = ["api", "ws_router", "JSONResponse"]
