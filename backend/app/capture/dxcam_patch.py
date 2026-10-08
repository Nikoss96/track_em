"""Точечное исправление dxcam 0.3.0: двойной Release COM-объектов.

DXGIDuplicator.release() и StageSurface.release() вызывают .Release() явно, затем присваивают None,
после чего comtypes.__del__ вызывает Release() повторно → use-after-free / access violation
при Stop. Оставляем один Release (через comtypes при сборке объекта).
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    from dxcam.core import dxgi_duplicator

    def release(self) -> None:  # type: ignore[no-untyped-def]
        if self.duplicator is not None:
            try:
                self.release_frame()
            except Exception:  # noqa: BLE001
                pass
            self.duplicator = None  # единственный Release — в comtypes.__del__

    def stage_release(self) -> None:  # type: ignore[no-untyped-def]
        if self.texture is not None:
            self.width = 0
            self.height = 0
            self.interface = None
            self.texture = None  # Release выполнит comtypes.__del__

    from dxcam.core import stagesurf

    dxgi_duplicator.DXGIDuplicator.release = release
    stagesurf.StageSurface.release = stage_release
    _applied = True
