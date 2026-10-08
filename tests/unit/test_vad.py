import asyncio

import numpy as np
import soundfile as sf
import soxr

from app.config import load_config
from app.preprocessing.vad import VadStage
from app.schemas.events import AudioChunk, EventType
from app.streaming.router import EventRouter


def _chunks(x: np.ndarray, gap_at: int | None = None):
    n = 3200
    for i in range(0, x.size - n + 1, n):
        yield AudioChunk(type=EventType.audio_chunk, session_id="s", t_ms=i // 16, pcm16k=x[i : i + n], duration_ms=200.0, seq=i // n,
                         source_rate=16000, source_channels=1, rms_dbfs=0.0, gap_before_ms=500.0 if gap_at == i // n else 0.0)


async def test_vad_segments_speech_not_silence(fixtures):
    x, sr = sf.read(str(fixtures / "reading_tts.wav"), dtype="float32")
    x = soxr.resample(x, sr, 16000)[: 16000 * 15]
    x = np.concatenate([np.zeros(16000 * 2, np.float32), x])  # 2 с тишины в начале
    r = EventRouter()
    sub = r.subscribe("t", [EventType.speech_activity], maxsize=1000)
    v = VadStage("s", r, load_config("cpu"))
    v.start()
    for ch in _chunks(x):
        v.feed(ch)
    v.flush()
    await v.stop()
    evs = []
    while (e := sub.get_nowait()) is not None:
        evs.append(e)
    ends = [e for e in evs if e.kind == "end"]
    assert len(ends) >= 2
    assert ends[0].t_ms >= 1900  # тишина не размечена как речь
    assert all(e.t_end_ms > e.t_ms for e in ends)
    assert all(a.t_end_ms <= b.t_ms for a, b in zip(ends, ends[1:], strict=False))


async def test_vad_silence_only(fixtures):
    r = EventRouter()
    sub = r.subscribe("t", [EventType.speech_activity], maxsize=100)
    v = VadStage("s", r, load_config("cpu"))
    v.start()
    for ch in _chunks(np.zeros(16000 * 5, np.float32)):
        v.feed(ch)
    v.flush()
    await v.stop()
    await asyncio.sleep(0)
    assert sub.get_nowait() is None
