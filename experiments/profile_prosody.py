"""Профилирование функций просодии на 4-секундном фрагменте TTS."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import soundfile as sf  # noqa: E402

from app.analyzers.prosody import features as F  # noqa: E402

x, sr = sf.read(Path(__file__).resolve().parents[1] / "tests/fixtures/reading_tts.wav", dtype="float32")
import soxr  # noqa: E402

x = soxr.resample(x, sr, 16000)[16000 * 4 : 16000 * 8]
for name, fn in [
    ("frame_db", lambda: F.frame_db(x)),
    ("f0", lambda: F.f0_contour(x, 75, 500)),
    ("egemaps_first", lambda: F.egemaps(x)),
    ("egemaps", lambda: F.egemaps(x)),
    ("rhythm", lambda: F.rhythm_stats(F.f0_contour(x, 75, 500), F.frame_db(x))),
]:
    t = time.perf_counter()
    fn()
    print(f"{name}: {(time.perf_counter() - t) * 1000:.0f} ms")
