"""API-контракты сессии."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionState(str, Enum):
    created = "created"
    initializing = "initializing"
    running = "running"
    stopping = "stopping"
    completed = "completed"
    failed = "failed"


class Region(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LiveSourceConfig(BaseModel):
    monitor: int = 0
    region: Region | None = None
    audio_device: str | None = None  # имя/ID выходного устройства для loopback
    fps: int = Field(default=30, ge=1, le=60)


class FileSourceConfig(BaseModel):
    path: str
    realtime: bool = True
    speed: float = Field(default=1.0, gt=0, le=16)


class SessionCreate(BaseModel):
    mode: Literal["reading", "qa"] = "reading"
    source: Literal["live", "file"] = "file"
    live: LiveSourceConfig | None = None
    file: FileSourceConfig | None = None
    reference_text: str | None = None
    question: str | None = None
    modules: list[str] | None = None  # None → из конфигурации
    profile: str | None = None  # cpu | gpu | None (auto)
    save_media: bool = False
    composite_enabled: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    id: str
    state: SessionState
    mode: str
    source: str
    created_at: str
    started_at: str | None = None
    stopped_at: str | None = None
    error: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
