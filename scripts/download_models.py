"""Предварительная загрузка открытых весов (этап установки; во время сессий сеть не нужна).

    .venv\\Scripts\\python.exe scripts\\download_models.py --profile gpu [--skip-llm]

Скачивает: MediaPipe face_landmarker.task, faster-whisper (модель профиля), emotion2vec+ (модель профиля),
и вызывает `ollama pull <llm.model>`, если Ollama установлен.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_config, models_dir, setup_cache_env  # noqa: E402
from app.storage.models import ensure_file  # noqa: E402

setup_cache_env()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--skip-llm", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.profile)
    print("профиль:", cfg["profile_resolved"])

    url = cfg["modules"]["face"]["model_url"]
    print("MediaPipe:", ensure_file(url, models_dir(cfg) / "mediapipe" / Path(url).name))

    from faster_whisper.utils import download_model

    name = cfg["modules"]["speech"]["model"]
    print("faster-whisper:", download_model(name, cache_dir=str(models_dir(cfg) / "whisper")))

    from huggingface_hub import snapshot_download

    ve = cfg["modules"]["voice_emotion"]["model"]
    print("emotion2vec:", snapshot_download(ve))

    if not a.skip_llm:
        if shutil.which("ollama"):
            subprocess.run(["ollama", "pull", cfg["llm"]["model"]], check=False)
        else:
            print("Ollama не найден: установите с https://ollama.com и выполните `ollama pull", cfg["llm"]["model"] + "`")


if __name__ == "__main__":
    main()
