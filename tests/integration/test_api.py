"""REST + WebSocket через TestClient."""

import csv
import io
import json
import time

from fastapi.testclient import TestClient

from app.main import app


def test_api_session_flow(fixtures):
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "ok"
        mods = c.get("/api/modules").json()
        assert {m["id"] for m in mods["modules"]} >= {"face", "prosody", "speech", "voice_emotion", "text", "transactional"}
        assert "mock" not in {m["id"] for m in mods["modules"]}
        d = c.get("/api/devices").json()
        assert "monitors" in d and "audio_outputs" in d

        assert c.post("/api/sessions", json={"source": "file", "file": {"path": "missing.mp4"}}).status_code == 400
        req = {"source": "file", "file": {"path": str(fixtures / "synthetic_av.mp4"), "realtime": True, "speed": 4}, "modules": ["mock"], "profile": "cpu"}
        s = c.post("/api/sessions", json=req).json()
        sid = s["id"]
        assert s["state"] == "created"
        with c.websocket_connect(f"/ws/sessions/{sid}") as ws:
            first = ws.receive_json()
            assert first["type"] == "session"
            r = c.post(f"/api/sessions/{sid}/start").json()
            assert r["state"] in ("initializing", "running")
            seen = set()
            t0 = time.time()
            while time.time() - t0 < 30:
                m = ws.receive_json()
                seen.add(m["type"])
                if m["type"] == "session" and m["data"]["state"] == "completed":
                    break
            assert {"status", "result", "preview"} <= seen
        res = c.get(f"/api/sessions/{sid}/results").json()
        assert res["session"]["state"] == "completed"
        assert any(r["module_id"] == "fusion" for r in res["results"])
        assert res["summary"]["scores"]["exploratory_composite_score"]["enabled"] is False
        filt = c.get(f"/api/sessions/{sid}/results", params={"module": "mock", "t_from": 2000, "t_to": 3000}).json()
        assert filt["results"] and all(2000 <= r["t_end_ms"] and r["t_start_ms"] <= 3000 for r in filt["results"])
        ej = c.get(f"/api/sessions/{sid}/export", params={"format": "json"})
        assert ej.status_code == 200 and json.loads(ej.content)["session"]["id"] == sid
        ec = c.get(f"/api/sessions/{sid}/export", params={"format": "csv"})
        rows = list(csv.DictReader(io.StringIO(ec.content.decode("utf-8"))))
        assert rows and {"module_id", "metric", "value_num", "t_start_ms"} <= set(rows[0])
