"""Event Router: типизированные подписки с ограниченными очередями.

Политики переполнения:
- drop_oldest — видео: устаревшие кадры выбрасываются;
- lossless    — аудио: при переполнении новый чанк не ставится, потерянная
                длительность явно добавляется в gap_before_ms следующего чанка;
- dedupe      — NLP: одинаковые ключи не обрабатываются повторно.

Все методы вызываются из потока event loop (источники используют
loop.call_soon_threadsafe).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from typing import Literal

from app.schemas.events import AudioChunk, Event, EventType

Policy = Literal["drop_oldest", "lossless", "dedupe"]
log = logging.getLogger(__name__)

_CLOSED = object()


class Subscription:
    def __init__(
        self,
        name: str,
        types: Iterable[EventType],
        maxsize: int,
        policy: Policy,
        key: Callable[[Event], Hashable] | None = None,
    ) -> None:
        self.name = name
        self.types = frozenset(types)
        self.maxsize = maxsize
        self.policy = policy
        self.key = key
        self._q: deque[object] = deque()
        self._ev = asyncio.Event()
        self._seen: deque[Hashable] = deque(maxlen=512)
        self._seen_set: set[Hashable] = set()
        self.delivered = 0
        self.dropped = 0
        self.deduped = 0
        self.lost_audio_ms = 0.0
        self._pending_gap_ms = 0.0
        self.closed = False

    def qsize(self) -> int:
        n = len(self._q)
        return n - 1 if n and self._q[-1] is _CLOSED else n

    def offer(self, ev: Event) -> None:
        if self.closed:
            return
        if self.policy == "dedupe" and self.key is not None:
            k = self.key(ev)
            if k in self._seen_set:
                self.deduped += 1
                return
            if len(self._seen) == self._seen.maxlen:
                self._seen_set.discard(self._seen[0])
            self._seen.append(k)
            self._seen_set.add(k)
        if len(self._q) >= self.maxsize:
            if self.policy == "drop_oldest" or self.policy == "dedupe":
                self._q.popleft()
                self.dropped += 1
            else:  # lossless
                self.dropped += 1
                if isinstance(ev, AudioChunk):
                    self._pending_gap_ms += ev.duration_ms
                    self.lost_audio_ms += ev.duration_ms
                return
        if self._pending_gap_ms and isinstance(ev, AudioChunk):
            ev = dataclasses.replace(ev, gap_before_ms=ev.gap_before_ms + self._pending_gap_ms)
            self._pending_gap_ms = 0.0
        self._q.append(ev)
        self.delivered += 1
        self._ev.set()

    def close(self) -> None:
        self.closed = True
        self._q.append(_CLOSED)
        self._ev.set()

    async def get(self) -> Event | None:
        while not self._q:
            self._ev.clear()
            await self._ev.wait()
        item = self._q.popleft()
        if item is _CLOSED:
            return None
        return item  # type: ignore[return-value]

    def get_nowait(self) -> Event | None:
        if not self._q or self._q[0] is _CLOSED:
            return None
        return self._q.popleft()  # type: ignore[return-value]

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> Event:
        ev = await self.get()
        if ev is None:
            raise StopAsyncIteration
        return ev

    def stats(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "policy": self.policy,
            "queue": len(self._q),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "deduped": self.deduped,
            "lost_audio_ms": round(self.lost_audio_ms, 1),
        }


class EventRouter:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self.published: dict[str, int] = {}

    def subscribe(
        self,
        name: str,
        types: Iterable[EventType],
        maxsize: int = 100,
        policy: Policy = "drop_oldest",
        key: Callable[[Event], Hashable] | None = None,
    ) -> Subscription:
        sub = Subscription(name, types, maxsize, policy, key)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.close()
        if sub in self._subs:
            self._subs.remove(sub)

    def saturated(self, etype: EventType, ratio: float = 0.75) -> bool:
        """Есть ли подписчик данного типа с очередью ≥ ratio·maxsize (для back-pressure офлайн-режима).
        Читается из потока источника: только чтение длины deque — безопасно."""
        for sub in list(self._subs):
            if etype in sub.types and not sub.closed and sub.qsize() >= max(1, int(sub.maxsize * ratio)):
                return True
        return False

    def publish(self, ev: Event) -> None:
        self.published[ev.type.value] = self.published.get(ev.type.value, 0) + 1
        for sub in self._subs:
            if ev.type in sub.types:
                sub.offer(ev)

    def close_all(self) -> None:
        for sub in list(self._subs):
            sub.close()
        self._subs.clear()

    def stats(self) -> dict[str, object]:
        return {"published": dict(self.published), "subscriptions": [s.stats() for s in self._subs]}
