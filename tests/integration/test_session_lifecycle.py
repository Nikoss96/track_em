"""Жизненный цикл сессии на File Replay + mock-анализаторе (без тяжёлых моделей)."""

from pathlib import Path

from app.config import data_dir, load_config
from app.schemas.session import FileSourceConfig, SessionCreate, SessionState
from app.session.manager import SessionManager
from app.storage.db import Database
from tests.conftest import wait_for


def _mgr() -> SessionManager:
    return SessionManager(Database(data_dir(load_config()) / "t.sqlite3"))


def _req(path: Path, **kw) -> SessionCreate:
    return SessionCreate(
        source="file", file=FileSourceConfig(path=str(path), realtime=False), modules=["mock"], profile="cpu", **kw
    )


async def test_full_lifecycle_file_replay(fixtures):
    mgr = _mgr()
    s = mgr.create(_req(fixtures / "synthetic_av.mp4"))
    assert s.state == SessionState.created
    states = []
    q = s.add_client()
    await s.start()
    await s.start()  # повторный Start — без дублирования
    assert await wait_for(lambda: s.state == SessionState.completed, timeout=60)  # EOF → авто-stop
    while not q.empty():
        m = q.get_nowait()
        if m["type"] == "session":
            states.append(m["data"]["state"])
    assert states[:2] == ["initializing", "running"]
    assert states[-2:] == ["stopping", "completed"]
    mock = [r for r in s.store.results if r.module_id == "mock"]
    frames = [r for r in mock if r.metric("frame_seq")]
    chunks = [r for r in mock if r.metric("rms_dbfs")]
    assert 300 <= len(frames) <= 362  # 12 c @ 30 fps
    # аудио: 12 c при 200 мс → ~60 чанков, временная ось монотонна
    assert 55 <= len(chunks) <= 62
    ts = [r.t_start_ms for r in chunks]
    assert ts == sorted(ts)
    # тишина 3–5 с видна в уровнях
    silent = [r for r in chunks if 3200 <= r.t_start_ms <= 4600]
    assert silent and all(r.metric("rms_dbfs").value < -60 for r in silent)
    # один source → один поток данных (нет дублирования при двойном start)
    seqs = [r.metric("frame_seq").value for r in frames]
    assert len(seqs) == len(set(seqs))
    # fusion окна и сохранение
    assert any(r.module_id == "fusion" for r in s.store.results)
    assert (s.dir / "summary.json").exists() and (s.dir / "metrics.parquet").exists()
    row = s.db.get_session(s.id)
    assert row["state"] == "completed" and row["stats"]["scores"]["groups"]


async def test_module_failure_isolated(fixtures):
    mgr = _mgr()
    req = _req(fixtures / "synthetic_av.mp4", overrides={"modules": {"mock": {"fail_on_seq": 5}}})
    s = mgr.create(req)
    await s.start()
    assert await wait_for(lambda: s.state == SessionState.completed, timeout=60)
    h = s.hosts["mock"].plugin.health()
    assert h.errors == 1 and h.processed > 300
    assert any(e["module_id"] == "mock" for e in s.module_errors)


async def test_user_stop_releases_and_saves(fixtures):
    mgr = _mgr()
    req = SessionCreate(source="file", file=FileSourceConfig(path=str(fixtures / "synthetic_av.mp4"), realtime=True), modules=["mock"], profile="cpu")
    s = mgr.create(req)
    await s.start()
    assert await wait_for(lambda: s.state == SessionState.running, timeout=30)
    assert await wait_for(lambda: s.processor.video_seq > 20, timeout=10)
    await s.stop()
    assert s.state == SessionState.completed
    assert not s.source.alive  # потоки источника завершены
    n = s.processor.video_seq
    assert n < 200  # остановлено раньше конца файла (realtime)


async def test_missing_file_fails_cleanly(tmp_path):
    mgr = _mgr()
    s = mgr.create(_req(tmp_path / "nope.mp4"))
    await s.start()
    assert await wait_for(lambda: s.state == SessionState.failed, timeout=30)
    assert "nope.mp4" in (s.error or "")
