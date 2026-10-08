import asyncio

import numpy as np

from app.config import load_config
from app.schemas.events import EventType
from app.streaming.clock import SessionClock
from app.streaming.processor import StreamProcessor
from app.streaming.router import EventRouter


async def _setup():
    loop = asyncio.get_running_loop()
    router = EventRouter()
    sub = router.subscribe("a", [EventType.audio_chunk, EventType.video_frame], maxsize=1000, policy="lossless")
    clock = SessionClock()
    clock.start()
    proc = StreamProcessor("s", router, loop, clock, load_config("cpu"))
    return router, sub, proc


async def test_audio_resampled_to_16k_mono_contiguous():
    _, sub, proc = await _setup()
    sr = 48000
    t = 0.0
    for _ in range(25):  # 25 × 100 мс стерео 48 кГц
        n = sr // 10
        x = np.stack([np.sin(np.arange(n) / sr * 2 * np.pi * 300)] * 2, axis=1).astype(np.float32) * 0.5
        proc.on_audio(x, sr, t)
        t += 100.0
    proc.finish()
    await asyncio.sleep(0.05)
    chunks = []
    while (ev := sub.get_nowait()) is not None:
        chunks.append(ev)
    assert len(chunks) >= 12
    assert all(c.pcm16k.dtype == np.float32 and c.pcm16k.ndim == 1 for c in chunks)
    assert chunks[0].source_rate == 48000 and chunks[0].source_channels == 2
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert abs(b.t_ms - a.t_end_ms) <= 1  # непрерывная временная ось
    total = sum(c.pcm16k.size for c in chunks)
    assert abs(total - 2.5 * 16000) < 400
    assert -12 < chunks[3].rms_dbfs < -6  # синус 0.5 → ≈ −9 dBFS


async def test_audio_gap_detected_and_marked():
    _, sub, proc = await _setup()
    sr = 16000
    proc.on_audio(np.zeros((3200, 1), np.float32), sr, 0.0)
    proc.on_audio(np.zeros((3200, 1), np.float32), sr, 1200.0)  # пропуск 1000 мс
    proc.finish()
    await asyncio.sleep(0.05)
    evs = []
    while (ev := sub.get_nowait()) is not None:
        evs.append(ev)
    assert proc.audio_gaps == 1
    assert any(e.gap_before_ms >= 999 for e in evs)
    assert evs[-1].t_ms == 1200


async def test_video_rgb_and_blank_detection():
    _, sub, proc = await _setup()
    proc.on_video(np.zeros((100, 100, 3), np.uint8), 0.0)
    noise = (np.random.default_rng(0).random((100, 100, 4)) * 255).astype(np.uint8)
    proc.on_video(noise, 33.0)  # BGRA → RGB
    await asyncio.sleep(0.05)
    a, b = sub.get_nowait(), sub.get_nowait()
    assert a.corrupted and not b.corrupted
    assert b.frame.shape == (100, 100, 3)
    assert proc.latest_preview is not None
