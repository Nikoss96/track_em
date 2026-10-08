"""Multimodal Behavior Analyzer."""

import sys

# DXcam (comtypes) и SoundCard используют COM. MTA-режим позволяет обращаться к
# DXGI/WASAPI-объектам из рабочих потоков захвата без access violation при освобождении.
if sys.platform == "win32":
    if not hasattr(sys, "coinit_flags"):
        sys.coinit_flags = 0  # COINIT_MULTITHREADED
    # SoundCard считает S_FALSE («COM уже инициализирован») ошибкой, поэтому
    # он должен инициализировать COM раньше comtypes (dxcam).
    try:
        import soundcard  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
