"""Перечисление мониторов и выходных аудиоустройств."""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)
_OUT_RE = re.compile(r"Device\[(\d+)\] Output\[(\d+)\]: Res:\((\d+), (\d+)\) Rot:(\d+) Primary:(\w+)")


def list_monitors() -> list[dict[str, Any]]:
    try:
        import dxcam

        info = dxcam.output_info()
    except Exception as e:  # noqa: BLE001
        log.warning("dxcam unavailable: %s", e)
        return []
    out = []
    for m in _OUT_RE.finditer(info):
        dev, idx, w, h, rot, prim = m.groups()
        out.append(
            {
                "device": int(dev),
                "index": int(idx),
                "width": int(w),
                "height": int(h),
                "rotation": int(rot),
                "primary": prim == "True",
            }
        )
    return out


def list_audio_outputs() -> list[dict[str, Any]]:
    try:
        import soundcard as sc

        default = sc.default_speaker()
        return [
            {
                "id": str(s.id),
                "name": s.name,
                "channels": s.channels,
                "default": default is not None and s.id == default.id,
            }
            for s in sc.all_speakers()
        ]
    except Exception as e:  # noqa: BLE001
        log.warning("soundcard unavailable: %s", e)
        return []
