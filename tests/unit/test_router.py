import numpy as np

from app.schemas.events import AudioChunk, EventType, Transcript, VideoFrame
from app.streaming.router import EventRouter


def _vf(seq: int) -> VideoFrame:
    return VideoFrame(type=EventType.video_frame, session_id="s", t_ms=seq * 33, frame=np.zeros((2, 2, 3), np.uint8), seq=seq)


def _ac(seq: int) -> AudioChunk:
    return AudioChunk(
        type=EventType.audio_chunk, session_id="s", t_ms=seq * 200, pcm16k=np.zeros(3200, np.float32),
        duration_ms=200.0, seq=seq, source_rate=48000, source_channels=2, rms_dbfs=-120.0,
    )


async def test_drop_oldest_keeps_latest_frames():
    r = EventRouter()
    sub = r.subscribe("v", [EventType.video_frame], maxsize=3, policy="drop_oldest")
    for i in range(10):
        r.publish(_vf(i))
    got = [sub.get_nowait().seq for _ in range(3)]
    assert got == [7, 8, 9]
    assert sub.dropped == 7


async def test_lossless_marks_gap_explicitly():
    r = EventRouter()
    sub = r.subscribe("a", [EventType.audio_chunk], maxsize=2, policy="lossless")
    for i in range(5):
        r.publish(_ac(i))
    a, b = sub.get_nowait(), sub.get_nowait()
    assert (a.seq, b.seq) == (0, 1)  # порядок сохранён
    r.publish(_ac(5))
    c = sub.get_nowait()
    assert c.seq == 5 and c.gap_before_ms == 600.0  # 3 потерянных чанка по 200 мс
    assert sub.lost_audio_ms == 600.0


async def test_dedupe_and_type_filter():
    r = EventRouter()
    sub = r.subscribe("t", [EventType.transcript_final], maxsize=10, policy="dedupe", key=lambda e: (e.segment_id, e.text))
    tr = lambda sid, txt: Transcript(type=EventType.transcript_final, session_id="s", t_ms=0, t_end_ms=1, text=txt, segment_id=sid)  # noqa: E731
    r.publish(tr(1, "привет"))
    r.publish(tr(1, "привет"))
    r.publish(tr(2, "мир"))
    r.publish(_vf(0))  # не подписан
    assert sub.qsize() == 2 and sub.deduped == 1


async def test_close_terminates_iteration():
    r = EventRouter()
    sub = r.subscribe("v", [EventType.video_frame])
    r.publish(_vf(1))
    sub.close()
    items = [e async for e in sub]
    assert [e.seq for e in items] == [1]
