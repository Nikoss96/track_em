"""Точка входа FastAPI: `uvicorn app.main:app` (cwd = backend/)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from app.config import ROOT, data_dir, load_config, setup_cache_env

setup_cache_env()

from datetime import UTC  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.api.routes import api, ws_router  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.session.manager import SessionManager  # noqa: E402
from app.storage.db import Database  # noqa: E402


def apply_retention(cfg: dict, db: Database) -> None:
    """Политика хранения: удалить каталоги и записи сессий старше storage.retention_days."""
    import shutil
    from datetime import datetime, timedelta

    days = int(cfg.get("storage", {}).get("retention_days", 0) or 0)
    if days <= 0:
        return
    limit = datetime.now(UTC) - timedelta(days=days)
    for row in db.list_sessions(limit=100000):
        try:
            created = datetime.fromisoformat(row["created_at"])
        except ValueError:
            continue
        if created < limit:
            shutil.rmtree(data_dir(cfg) / "sessions" / row["id"], ignore_errors=True)
            db.delete_session(row["id"])


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    setup_logging()
    cfg = load_config()
    db = Database(data_dir(cfg) / "mba.sqlite3")
    apply_retention(cfg, db)
    app.state.db = db
    app.state.manager = SessionManager(db)
    yield
    await app.state.manager.shutdown()
    db.close()


app = FastAPI(title="Multimodal Behavior Analyzer", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api)
app.include_router(ws_router)

_dist = Path(ROOT) / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")
