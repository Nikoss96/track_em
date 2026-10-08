"""FaceAnalyzer на реальной модели MediaPipe (скачивается при первом запуске, ~4 МБ)."""

import pytest

from tests.harness import Harness

pytestmark = pytest.mark.slow


async def test_face_windows_presence_and_absence(fixtures):
    h = Harness(["face"])
    await h.run_file(fixtures / "reading_av.mp4", max_duration_s=16, with_vad=False)
    rs = sorted(h.by("face"), key=lambda r: r.t_start_ms)
    assert len(rs) >= 14
    # окна по 1 с, непрерывная ось
    assert all(r.t_end_ms - r.t_start_ms == 1000 for r in rs)
    with_face = [r for r in rs if r.t_end_ms <= 10000]
    no_face = [r for r in rs if 10200 <= r.t_start_ms and r.t_end_ms <= 12800]
    assert all(r.metric("face_presence").value >= 0.9 for r in with_face)
    assert no_face and all(r.status.value == "insufficient_data" for r in no_face)
    assert all(r.metric("smile") is None for r in no_face)  # без лица — нет мимических метрик
    r = with_face[3]
    assert 0 <= r.metric("expression_intensity").value <= 1
    assert r.metric("smile").value > 0.3  # на портрете выраженная улыбка
    assert len(r.payload["blendshapes_mean"]) >= 50
    assert h.models and h.models[0]["sha256"]
