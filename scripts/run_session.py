"""Прогон сессии через REST API (backend должен быть запущен).

    .venv\\Scripts\\python.exe scripts\\run_session.py --file tests\\fixtures\\reading_av.mp4 --reference tests\\fixtures\\reading_reference.txt
    .venv\\Scripts\\python.exe scripts\\run_session.py --live --seconds 30 --region 0,0,1280,720
    опции: --modules face,prosody,speech --mode qa --question "..." --fast (не в реальном времени) --no-wait

Печатает ID сессии, ход выполнения и итоговую сводку; экспорт — /api/sessions/<id>/export.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--file")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seconds", type=float, default=30, help="длительность Live-сессии")
    ap.add_argument("--region", help="left,top,width,height")
    ap.add_argument("--monitor", type=int, default=0)
    ap.add_argument("--audio-device")
    ap.add_argument("--reference")
    ap.add_argument("--mode", default="reading", choices=["reading", "qa"])
    ap.add_argument("--question")
    ap.add_argument("--modules")
    ap.add_argument("--profile")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--composite", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    body: dict = {"mode": a.mode, "source": "live" if a.live else "file", "composite_enabled": a.composite, "profile": a.profile}
    if a.live:
        live: dict = {"monitor": a.monitor, "audio_device": a.audio_device}
        if a.region:
            x, y, w, h = (int(v) for v in a.region.split(","))
            live["region"] = {"left": x, "top": y, "width": w, "height": h}
        body["live"] = live
    else:
        body["file"] = {"path": str(Path(a.file).resolve()), "realtime": not a.fast}
    if a.reference:
        body["reference_text"] = Path(a.reference).read_text(encoding="utf-8")
    if a.question:
        body["question"] = a.question
    if a.modules:
        body["modules"] = a.modules.split(",")

    c = httpx.Client(base_url=a.api, timeout=30)
    s = c.post("/api/sessions", json=body)
    if s.status_code != 200:
        print("ошибка:", s.text)
        return 1
    sid = s.json()["id"]
    print("session", sid)
    c.post(f"/api/sessions/{sid}/start").raise_for_status()
    if a.no_wait:
        return 0
    t0 = time.time()
    while True:
        st = c.get(f"/api/sessions/{sid}").json()
        state = st.get("state")
        status = st.get("status") or {}
        mods = {m["module_id"]: m["state"] for m in status.get("modules", []) if m["state"] != "disabled"}
        print(f"\r{time.time() - t0:6.1f}s {state:12} media={status.get('media_ms', 0) / 1000:6.1f}s results={status.get('results', 0):5} {mods}", end="", flush=True)
        if state in ("completed", "failed"):
            break
        if a.live and state == "running" and time.time() - t0 > a.seconds:
            c.post(f"/api/sessions/{sid}/stop")
        time.sleep(1)
    print()
    res = c.get(f"/api/sessions/{sid}/results").json()
    summ = res.get("summary", {})
    by_mod: dict[str, dict[str, int]] = {}
    for r in res["results"]:
        by_mod.setdefault(r["module_id"], {}).setdefault(r["status"], 0)
        by_mod[r["module_id"]][r["status"]] += 1
    print(json.dumps({"state": res["session"]["state"], "error": res["session"].get("error"), "results": by_mod,
                      "modules": [(m["module_id"], m["state"], m.get("detail")) for m in summ.get("modules", [])],
                      "capture": {k: summ.get("capture", {}).get(k) for k in ("video", "audio")}}, ensure_ascii=False, indent=1))
    print(f"export: {a.api}/api/sessions/{sid}/export?format=json | csv")
    return 0 if res["session"]["state"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
