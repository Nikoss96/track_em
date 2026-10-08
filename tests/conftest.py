from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MBA_DATA_DIR", str(tmp_path / "data"))
    yield


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


async def wait_for(pred, timeout: float = 30.0, interval: float = 0.05) -> bool:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(interval)
    return pred()


os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
