"""Генерация синтетических тестовых медиафайлов через ffmpeg (tests/fixtures).

- synthetic_av.mp4: 12 c, testsrc2 видео 640x360@30 + тон 220 Гц с паузами (тишина 3–5 c и 8–9 c);
- silence.wav: 5 c тишины;
- tone.wav: 3 c, тон 200 Гц.
Синтетика проверяет инфраструктуру (время, очереди, тишина), а не качество моделей.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def run(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    vol = "volume='if(between(t,3,5)+between(t,8,9),0,1)':eval=frame"
    run([
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=12",
        "-filter_complex", f"[1:a]{vol},aformat=channel_layouts=stereo[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(FIX / "synthetic_av.mp4"),
    ])
    run(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", str(FIX / "silence.wav")])
    run(["-f", "lavfi", "-i", "sine=frequency=200:sample_rate=48000:duration=3", str(FIX / "tone.wav")])
    print("fixtures:", sorted(p.name for p in FIX.iterdir()))


if __name__ == "__main__":
    main()
