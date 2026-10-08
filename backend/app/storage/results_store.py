"""In-memory хранилище результатов сессии + Parquet/JSON/CSV экспорт."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.schemas.results import AnalysisResult

TS_COLUMNS = [
    "session_id",
    "module_id",
    "model_id",
    "model_version",
    "t_start_ms",
    "t_end_ms",
    "finality",
    "result_status",
    "metric",
    "value_num",
    "value_str",
    "unit",
    "normalized_score",
    "confidence",
    "method",
    "metric_status",
    "kind",
    "group",
]


class ResultStore:
    """Хранит результаты; provisional-результат заменяется финальным по payload.replace_key."""

    def __init__(self) -> None:
        self.results: list[AnalysisResult] = []
        self._by_key: dict[str, int] = {}
        self.raw_frames: dict[str, list[dict[str, Any]]] = {}  # module_id → сырые записи

    def add(self, r: AnalysisResult) -> bool:
        """True — новый результат, False — замена существующего."""
        raw = r.payload.pop("raw", None)
        if raw:
            self.raw_frames.setdefault(r.module_id, []).extend(raw)
        key = r.payload.get("replace_key")
        if key is not None:
            k = f"{r.module_id}:{key}"
            idx = self._by_key.get(k)
            if idx is not None:
                self.results[idx] = r
                return False
            self._by_key[k] = len(self.results)
        self.results.append(r)
        return True

    def by_module(self, module_id: str) -> list[AnalysisResult]:
        return [r for r in self.results if r.module_id == module_id]

    def dicts(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.results]


def metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        for m in r.get("metrics", []):
            v = m.get("value")
            num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            if isinstance(v, bool):
                num = float(v)
            sval = None if num is not None else (json.dumps(v, ensure_ascii=False) if v is not None else None)
            rows.append(
                {
                    "session_id": r["session_id"],
                    "module_id": r["module_id"],
                    "model_id": r["model_id"],
                    "model_version": r["model_version"],
                    "t_start_ms": r["t_start_ms"],
                    "t_end_ms": r["t_end_ms"],
                    "finality": r.get("finality", "final"),
                    "result_status": r["status"],
                    "metric": m["name"],
                    "value_num": num,
                    "value_str": sval,
                    "unit": m.get("unit"),
                    "normalized_score": m.get("normalized_score"),
                    "confidence": m.get("confidence"),
                    "method": m.get("method"),
                    "metric_status": m.get("status"),
                    "kind": m.get("kind"),
                    "group": m.get("group"),
                }
            )
    return rows


def write_parquet(results: list[dict[str, Any]], raw_frames: dict[str, list[dict[str, Any]]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    rows = metric_rows(results)
    if rows:
        table = pa.Table.from_pylist(rows)
        p = out_dir / "metrics.parquet"
        pq.write_table(table, p)
        written.append(str(p))
    for module_id, recs in raw_frames.items():
        if not recs:
            continue
        flat = [_flatten(r) for r in recs]
        keys: dict[str, None] = {}
        for r in flat:
            keys.update(dict.fromkeys(r))
        flat = [{k: r.get(k) for k in keys} for r in flat]
        p = out_dir / f"raw_{module_id}.parquet"
        pq.write_table(pa.Table.from_pylist(flat), p)
        written.append(str(p))
    return written


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def to_csv(results: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TS_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for row in metric_rows(results):
        w.writerow(row)
    return buf.getvalue()
