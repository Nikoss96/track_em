"""Фикстура Reading Mode: русская речь (Windows SAPI, голос Irina) по известному эталону
(А. П. Чехов, «Толстый и тонкий», общественное достояние) + видео с лицом.

Видео: портрет из тестовых ассетов MediaPipe (Apache-2.0) с медленным zoom/pan;
на 10–13 c кадр закрыт (проверка «лицо отсутствует»).
Синтетическая речь не несёт естественной эмоциональной экспрессии — фикстура проверяет
работоспособность pipeline, а не качество/валидность моделей.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "backend"))

TEXT = (
    "На вокзале Николаевской железной дороги встретились два приятеля: один толстый, другой тонкий. "
    "Толстый только что пообедал на вокзале, и губы его, подёрнутые маслом, лоснились, как спелые вишни. "
    "Пахло от него хересом и флёр-д'оранжем. "
    "Тонкий же только что вышел из вагона и был навьючен чемоданами, узлами и картонками.\n"
    "— Порфирий! — воскликнул толстый, увидев тонкого. — Ты ли это? Голубчик мой! Сколько зим, сколько лет!\n"
    "— Батюшки! — изумился тонкий. — Миша! Друг детства! Откуда ты взялся?\n"
)
PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"


def tts(text: str, out: Path, voice: str = "Microsoft Irina Desktop", rate: int = 0) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); $s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{out}'); "
        f"$s.Speak([IO.File]::ReadAllText('{out.with_suffix('.txt')}', [Text.Encoding]::UTF8)); $s.Dispose()"
    )
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    out.with_suffix(".txt").unlink()


def main() -> None:
    from app.storage.models import ensure_file

    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "reading_reference.txt").write_text(TEXT, encoding="utf-8")
    wav = FIX / "reading_tts.wav"
    tts(TEXT.replace("—", ",").replace("\n", " "), wav)
    img = ensure_file(PORTRAIT_URL, FIX / "portrait.jpg")
    dur = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    frames = int(dur * 25) + 25
    vf = (
        f"scale=1280:-2,zoompan=z='1+0.0015*on':x='iw/2-(iw/zoom/2)+20*sin(on/40)':y='ih/2-(ih/zoom/2)':d={frames}:s=640x640:fps=25,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,10,13)',format=yuv420p"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(img), "-i", str(wav), "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-b:a", "128k", "-shortest", str(FIX / "reading_av.mp4")],
        check=True,
    )
    print(f"reading_av.mp4: {dur:.1f} s; reference: {len(TEXT.split())} words")


if __name__ == "__main__":
    main()
