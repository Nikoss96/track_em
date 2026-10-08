"""SQLite: сессии, конфигурации, статусы, результаты (JSON), реестр моделей."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    mode TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    stopped_at TEXT,
    error TEXT,
    request_json TEXT,
    config_json TEXT,
    stats_json TEXT
);
CREATE TABLE IF NOT EXISTS results (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    module_id TEXT NOT NULL,
    t_start_ms INTEGER NOT NULL,
    t_end_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    finality TEXT NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_results_session_module ON results(session_id, module_id, t_start_ms);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    backend TEXT NOT NULL,
    source TEXT,
    license TEXT,
    languages TEXT,
    local_path TEXT,
    sha256 TEXT,
    registered_at TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (model_id, version)
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- sessions ---------------------------------------------------------
    def insert_session(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id,state,mode,source,created_at,request_json,config_json) VALUES (?,?,?,?,?,?,?)",
                (
                    row["id"],
                    row["state"],
                    row["mode"],
                    row["source"],
                    row["created_at"],
                    json.dumps(row.get("request", {}), ensure_ascii=False),
                    json.dumps(row.get("config", {}), ensure_ascii=False, default=str),
                ),
            )
            self._conn.commit()

    def update_session(self, sid: str, **fields: Any) -> None:
        cols, vals = [], []
        for k, v in fields.items():
            if k in ("stats", "config"):
                k, v = f"{k}_json", json.dumps(v, ensure_ascii=False, default=str)
            cols.append(f"{k}=?")
            vals.append(v)
        if not cols:
            return
        with self._lock:
            self._conn.execute(f"UPDATE sessions SET {','.join(cols)} WHERE id=?", (*vals, sid))
            self._conn.commit()

    def get_session(self, sid: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM sessions WHERE id=?", (sid,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
        return _decode_session(dict(zip(cols, row, strict=True)))

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id,state,mode,source,created_at,started_at,stopped_at,error FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def delete_session(self, sid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM results WHERE session_id=?", (sid,))
            self._conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            self._conn.commit()

    # --- results ----------------------------------------------------------
    def replace_results(self, sid: str, results: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM results WHERE session_id=?", (sid,))
            self._conn.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        sid,
                        i,
                        r["module_id"],
                        r["t_start_ms"],
                        r["t_end_ms"],
                        r["status"],
                        r.get("finality", "final"),
                        json.dumps(r, ensure_ascii=False, default=str),
                    )
                    for i, r in enumerate(results)
                ],
            )
            self._conn.commit()

    def load_results(self, sid: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT json FROM results WHERE session_id=? ORDER BY seq", (sid,))
            return [json.loads(r[0]) for r in cur.fetchall()]

    # --- models -----------------------------------------------------------
    def register_model(self, **m: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    m["model_id"],
                    m.get("version", "unknown"),
                    m.get("backend", ""),
                    m.get("source"),
                    m.get("license"),
                    m.get("languages"),
                    m.get("local_path"),
                    m.get("sha256"),
                    now_iso(),
                    m.get("notes"),
                ),
            )
            self._conn.commit()

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM models ORDER BY model_id")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _decode_session(row: dict[str, Any]) -> dict[str, Any]:
    for k in ("request_json", "config_json", "stats_json"):
        v = row.pop(k, None)
        row[k[:-5]] = json.loads(v) if v else {}
    return row
