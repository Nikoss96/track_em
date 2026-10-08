"""Проверка Phase 7 на целевой машине (RTX 3080 / i9-12900KF / 32 GB).

    .venv\\Scripts\\python.exe scripts\\verify_target.py [--file D:\\video.mp4 --reference text.txt] [--live-seconds 20]

Шаги:
 1. check_env (CUDA, Ollama, устройства);
 2. запуск backend (подпроцесс), полная Reading-сессия File Replay (по умолчанию склейка фикстуры ≥ 100 с)
    со ВСЕМИ модулями и реальными моделями; опрос /api/health → пики RSS и VRAM;
 3. повторный прогон того же файла в ускоренном режиме → сравнение измерений (воспроизводимость);
 4. Live-сессия (экран + loopback) заданной длительности;
 5. отчёт data/verification_report.json и краткая таблица в консоли.
Ничего не отправляется во внешние сервисы.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
API = "http://127.0.0.1:8000"


def start_backend() -> subprocess.Popen:
    (ROOT / "data").mkdir(exist_ok=True)
    log = open(ROOT / "data" / "verify_backend.log", "w", encoding="utf-8")
    p = subprocess.Popen([PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT / "backend", stdout=log, stderr=log)
    for _ in range(120):
        try:
            httpx.get(f"{API}/api/health", timeout=1)
            return p
        except Exception:  # noqa: BLE001
            time.sleep(1)
    p.kill()
    raise RuntimeError("backend не стартовал, см. data/verify_backend.log")


class Monitor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.peak_rss = 0
        self.peak_vram = 0
        self.samples = 0

    def run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                h = httpx.get(f"{API}/api/health", timeout=5).json()
                self.peak_rss = max(self.peak_rss, h["process_rss_mb"])
                for g in h.get("gpus", []):
                    self.peak_vram = max(self.peak_vram, g["memory_used_mb"])
                self.samples += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)


def run_session(body: dict, live_seconds: float | None = None, timeout: float = 1800) -> dict:
    c = httpx.Client(base_url=API, timeout=60)
    sid = c.post("/api/sessions", json=body).raise_for_status().json()["id"]
    c.post(f"/api/sessions/{sid}/start").raise_for_status()
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/sessions/{sid}").json()
        if live_seconds and st["state"] == "running" and time.time() - t0 > live_seconds:
            c.post(f"/api/sessions/{sid}/stop")
        if st["state"] in ("completed", "failed"):
            break
        time.sleep(2)
    d = c.get(f"/api/sessions/{sid}/results").json()
    s = d["summary"]
    lat: dict[str, list[float]] = {}
    for r in d["results"]:
        if r.get("latency_ms"):
            lat.setdefault(r["module_id"], []).append(r["latency_ms"])
    return {
        "id": sid,
        "state": d["session"]["state"],
        "error": d["session"].get("error"),
        "wall_s": round(time.time() - t0, 1),
        "statuses": {f"{k[0]}:{k[1]}": v for k, v in Counter((r["module_id"], r["status"]) for r in d["results"]).items()},
        "modules": {m["module_id"]: {"state": m["state"], "model": m.get("model_id"), "detail": m.get("detail"), "avg_latency_ms": m.get("avg_latency_ms"), "dropped": m.get("dropped")} for m in s.get("modules", [])},
        "latency_p95_ms": {k: round(sorted(v)[int(0.95 * (len(v) - 1))], 1) for k, v in lat.items()},
        "capture": {k: s.get("capture", {}).get(k) for k in ("video", "audio")},
        "reading": next((i for i in s.get("scores", {}).get("groups", {}).get("reading_accuracy", [])), None),
        "results": d["results"],
    }


def snapshot(results: list[dict]) -> list:
    out = []
    for r in results:
        if r["module_id"] in ("face", "prosody", "speech"):
            ms = tuple((m["name"], round(m["value"], 4) if isinstance(m["value"], float) else m["value"]) for m in r["metrics"] if isinstance(m["value"], (int, float, str)))
            out.append((r["module_id"], r["t_start_ms"], r["t_end_ms"], ms))
    return sorted(out, key=lambda x: (x[0], x[1], x[2], str(x[3])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--reference")
    ap.add_argument("--live-seconds", type=float, default=20)
    a = ap.parse_args()
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    subprocess.run([PY, str(ROOT / "scripts" / "check_env.py")], check=False)

    media = Path(a.file) if a.file else ROOT / "data" / "reading_av_100s.mp4"
    if not a.file and not media.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "1", "-i", str(ROOT / "tests/fixtures/reading_av.mp4"), "-c", "copy", str(media)], check=True)
    ref = Path(a.reference).read_text(encoding="utf-8") if a.reference else (ROOT / "tests/fixtures/reading_reference.txt").read_text(encoding="utf-8")

    p = start_backend()
    mon = Monitor()
    mon.start()
    try:
        print("\n== 1. File Replay (реальное время, все модули)")
        r1 = run_session({"mode": "reading", "source": "file", "file": {"path": str(media), "realtime": True}, "reference_text": ref})
        print("\n== 2. Повтор (ускоренно) — воспроизводимость")
        r2 = run_session({"mode": "reading", "source": "file", "file": {"path": str(media), "realtime": False}, "reference_text": ref})
        r3 = run_session({"mode": "reading", "source": "file", "file": {"path": str(media), "realtime": False}, "reference_text": ref})
        s2, s3 = snapshot(r2["results"]), snapshot(r3["results"])
        report["reproducible"] = {"identical": s2 == s3, "n": len(s2), "diff": sum(1 for x, y in zip(s2, s3, strict=False) if x != y) + abs(len(s2) - len(s3))}
        print(f"\n== 3. Live {a.live_seconds:.0f} c (экран + системный звук)")
        r4 = run_session({"mode": "reading", "source": "live", "live": {"monitor": 0}}, live_seconds=a.live_seconds)
    finally:
        mon.stop_ev.set()
        p.terminate()
    for k, r in (("file_realtime", r1), ("file_fast_a", r2), ("file_fast_b", r3), ("live", r4)):
        r.pop("results", None)
        report[k] = r
    report["peak_rss_mb"] = mon.peak_rss
    report["peak_vram_mb"] = mon.peak_vram
    out = ROOT / "data" / "verification_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n== ИТОГ")
    for k in ("file_realtime", "live"):
        r = report[k]
        print(f"{k:14} {r['state']:10} {r['wall_s']:7.1f} c  ошибка={r['error']}")
        for m, v in r["modules"].items():
            print(f"    {m:14} {v['state']:12} p95={r['latency_p95_ms'].get(m, '—')} мс  dropped={v['dropped']}  {v['detail'] or ''}")
    print(f"воспроизводимость: {report['reproducible']}")
    print(f"пик RSS {mon.peak_rss} МБ, пик VRAM {mon.peak_vram} МБ")
    if report["file_realtime"]["reading"]:
        print("reading_accuracy:", report["file_realtime"]["reading"].get("raw"))
    print("отчёт:", out)
    ok = all(report[k]["state"] == "completed" for k in ("file_realtime", "file_fast_a", "file_fast_b", "live"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
