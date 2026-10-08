"""Проверка окружения: Python, CUDA (CTranslate2/torch), VRAM, ffmpeg, Ollama, устройства захвата, диск.

    .venv\\Scripts\\python.exe scripts\\check_env.py
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import app  # noqa: E402,F401  (COM init order)
from app.config import gpu_memory, load_config, setup_cache_env  # noqa: E402

setup_cache_env()


def row(name: str, ok: bool | None, detail: str) -> None:
    mark = {True: "OK  ", False: "FAIL", None: "WARN"}[ok]
    print(f"[{mark}] {name:28} {detail}")


def main() -> None:
    cfg = load_config()
    row("Python", sys.version_info[:2] == (3, 11), platform.python_version())
    row("Профиль", None if cfg["profile_resolved"] == "cpu" else True, cfg["profile_resolved"])
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        row("CTranslate2 CUDA", n > 0 or None, f"{ctranslate2.__version__}, устройств: {n}")
        if n:
            types = ctranslate2.get_supported_compute_types("cuda")
            row("CT2 compute types", "int8_float16" in types, ", ".join(sorted(types)))
    except Exception as e:  # noqa: BLE001
        row("CTranslate2", False, str(e))
    try:
        import torch

        cuda = torch.cuda.is_available()
        row("torch", cuda or None, f"{torch.__version__}, cuda={cuda}" + (f", {torch.cuda.get_device_name(0)}" if cuda else ""))
    except Exception as e:  # noqa: BLE001
        row("torch", None, f"не установлен: {e}")
    gpus = gpu_memory()
    row("nvidia-smi", bool(gpus) or None, "; ".join(f"{g['name']} {g['memory_used_mb']}/{g['memory_total_mb']} МБ" for g in gpus) or "не найден")
    row("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "не найден в PATH")
    if cfg["llm"].get("backend") == "openai":
        import httpx

        from app.analyzers.llm_client import read_openai_key

        key = read_openai_key()
        model = cfg["llm"].get("openai_model")
        if not cfg["llm"].get("allow_external"):
            row("OpenAI", False, "llm.allow_external = false — включите в «Настройках»")
        elif not key:
            row("OpenAI", False, "нет ключа (OPENAI_API_KEY или «Настройки»)")
        else:
            try:
                r = httpx.get(f"{cfg['llm'].get('openai_url', 'https://api.openai.com/v1')}/models/{model}", headers={"Authorization": f"Bearer {key}"}, timeout=10)
                row("OpenAI", r.status_code == 200, f"{model}: HTTP {r.status_code}; тариф {cfg['llm'].get('pricing', {}).get(model, 'не задан')}")
            except Exception as e:  # noqa: BLE001
                row("OpenAI", False, f"нет связи: {type(e).__name__}")
        return_ollama = False
    else:
        return_ollama = True
    try:
        import httpx

        if not return_ollama:
            raise RuntimeError("skip")
        url = cfg["llm"]["url"]
        tags = httpx.get(f"{url}/api/tags", timeout=3).json().get("models", [])
        names = [m["name"] for m in tags]
        want = cfg["llm"]["model"]
        row("Ollama", want in names or f"{want}:latest" in names, f"{url}; модели: {names}; нужна {want}")
    except Exception as e:  # noqa: BLE001
        if return_ollama:
            row("Ollama", False, f"не отвечает ({type(e).__name__}); TextAnalyzer/TransactionalAnalyzer будут unavailable")
    from app.capture.devices import list_audio_outputs, list_monitors

    mons = list_monitors()
    row("Мониторы (DXcam)", bool(mons), ", ".join(f"#{m['index']} {m['width']}x{m['height']}" for m in mons) or "нет")
    outs = list_audio_outputs()
    row("Аудиовыходы (loopback)", bool(outs), "; ".join(o["name"] + (" *" if o["default"] else "") for o in outs) or "нет")
    for d in {ROOT.drive or "/", "C:\\"}:
        try:
            free = shutil.disk_usage(d).free / 2**30
            row(f"Диск {d}", free > 10 or None, f"свободно {free:.1f} ГБ")
        except OSError:
            pass
    import psutil

    vm = psutil.virtual_memory()
    row("RAM", vm.total / 2**30 >= 15 or None, f"{vm.total / 2**30:.1f} ГБ, свободно {vm.available / 2**30:.1f} ГБ")


if __name__ == "__main__":
    main()
