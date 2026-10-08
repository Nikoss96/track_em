"""Загрузка конфигурации: default.yaml + профиль + overrides сессии."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def resolve_profile(name: str | None) -> str:
    if name in (None, "", "auto"):
        return "gpu" if cuda_available() else "cpu"
    return name


def load_config(profile: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    user_path = CONFIG_DIR / "user.yaml"  # изменения из UI Settings
    if user_path.exists():
        with open(user_path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    prof = resolve_profile(profile or cfg.get("profile"))
    prof_path = CONFIG_DIR / "profiles" / f"{prof}.yaml"
    if prof_path.exists():
        with open(prof_path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg["profile_resolved"] = prof
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def save_user_config(patch: dict[str, Any]) -> dict[str, Any]:
    user_path = CONFIG_DIR / "user.yaml"
    current: dict[str, Any] = {}
    if user_path.exists():
        with open(user_path, encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}
    merged = deep_merge(current, patch)
    with open(user_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
    return merged


SECRETS_PATH = CONFIG_DIR / "secrets.yaml"  # в .gitignore; ключи никогда не возвращаются через API


def read_secrets() -> dict[str, Any]:
    if not SECRETS_PATH.exists():
        return {}
    with open(SECRETS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_secrets(patch: dict[str, Any]) -> None:
    cur = read_secrets()
    for k, v in patch.items():
        if v in (None, ""):
            cur.pop(k, None)
        else:
            cur[k] = v
    with open(SECRETS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cur, f, allow_unicode=True)


def data_dir(cfg: dict[str, Any]) -> Path:
    p = Path(os.environ.get("MBA_DATA_DIR", cfg["app"]["data_dir"]))
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["app"]["models_dir"])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_cache_env() -> None:
    """Кэши моделей — внутри проекта (на машине мало места на C:)."""
    cache = ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache / "modelscope"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    setup_cuda_dll_paths()


def setup_cuda_dll_paths() -> list[str]:
    """Windows: CTranslate2 (faster-whisper) ищет cublas64_12.dll / cudnn64_9.dll в PATH.
    Добавляем каталоги DLL из pip-пакетов nvidia-* и из CUDA-сборки torch, если они установлены."""
    import sys
    import sysconfig

    if sys.platform != "win32":
        return []
    site = Path(sysconfig.get_paths()["purelib"])
    dirs = [p for p in (site / "nvidia").glob("*/bin") if p.is_dir()]
    if (site / "torch" / "lib").is_dir():
        dirs.append(site / "torch" / "lib")
    added = []
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(d))
        except OSError:
            pass
    return added


def gpu_memory() -> list[dict[str, Any]]:
    """Использование VRAM через nvidia-smi (без зависимости от torch)."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    res = []
    for line in out.strip().splitlines():
        name, used, total, util = (x.strip() for x in line.split(","))
        res.append({"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization_pct": int(util)})
    return res
