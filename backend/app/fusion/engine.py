"""TemporalFusionEngine — late fusion по скользящим окнам на общей оси времени.

- окно window_ms (5 с), шаг step_ms (1 с);
- каждый результат имеет интервал [t_start, t_end]; вклад в окно взвешивается перекрытием;
- поздние результаты (ASR/NLP) помечают затронутые окна «грязными» → пересчёт;
- окно provisional, пока есть provisional-входы или ожидаются запаздывающие модальности;
- отсутствующие модальности фиксируются явно (availability).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from app.fusion.crossmodal import window_alignment
from app.schemas.results import AnalysisResult, Metric, ResultStatus

LATE_MODULES = {"speech", "text", "transactional", "voice_emotion"}


class TemporalFusionEngine:
    module_id = "fusion"
    version = "0.1.0"

    def __init__(self, session_id: str, cfg: dict[str, Any], modules: list[str]) -> None:
        f = cfg["fusion"]
        self.session_id = session_id
        self.window = int(f["window_ms"])
        self.step = int(f["step_ms"])
        self.late_horizon = int(f["late_horizon_ms"])
        self.modules = [m for m in modules if m not in ("fusion", "mock")]
        self._by_module: dict[str, list[AnalysisResult]] = defaultdict(list)
        self._last_end: dict[str, int] = {}
        self.computed: dict[int, AnalysisResult] = {}
        self._dirty: set[int] = set()
        self._next_start = 0  # следующий ещё не вычислявшийся старт окна
        self.recomputed = 0

    # --- вход -------------------------------------------------------------
    def add(self, r: AnalysisResult) -> None:
        if r.module_id in ("fusion", "scoring") or r.payload.get("summary"):
            return
        if r.t_end_ms < r.t_start_ms:
            return
        lst = self._by_module[r.module_id]
        key = r.payload.get("replace_key")
        if key is not None:
            for i, old in enumerate(lst):
                if old.payload.get("replace_key") == key:
                    lst[i] = r
                    break
            else:
                lst.append(r)
        else:
            lst.append(r)
        self._last_end[r.module_id] = max(self._last_end.get(r.module_id, 0), r.t_end_ms)
        # поздний результат: пометить уже вычисленные окна, которые он перекрывает
        for ws in self.computed:
            if ws < r.t_end_ms and ws + self.window > r.t_start_ms:
                self._dirty.add(ws)

    # --- вычисление -------------------------------------------------------
    def tick(self, now_ms: float) -> list[AnalysisResult]:
        """Новые окна, полностью завершившиеся к now_ms, плюс пересчитанные грязные."""
        out: list[AnalysisResult] = []
        while self._next_start + self.window <= now_ms:
            ws = self._next_start
            out.append(self._compute(ws, now_ms, final=False))
            self._next_start += self.step
        for ws in sorted(self._dirty):
            if ws in self.computed:
                self.recomputed += 1
                out.append(self._compute(ws, now_ms, final=False))
        self._dirty.clear()
        # окна старше late_horizon становятся финальными
        for ws, r in self.computed.items():
            if r.finality == "provisional" and now_ms - (ws + self.window) > self.late_horizon:
                out.append(self._compute(ws, now_ms, final=True))
        return out

    def finalize(self, end_ms: float) -> list[AnalysisResult]:
        out = []
        while self._next_start < end_ms:
            out.append(self._compute(self._next_start, end_ms, final=True))
            self._next_start += self.step
        for ws in list(self.computed):
            if self.computed[ws].finality != "final" or ws in self._dirty:
                out.append(self._compute(ws, end_ms, final=True))
        self._dirty.clear()
        return out

    def _compute(self, ws: int, now_ms: float, final: bool) -> AnalysisResult:
        we = ws + self.window
        metrics: list[Metric] = []
        availability: dict[str, str] = {}
        provisional_inputs = False
        fused: dict[str, Any] = {}
        for mod in self.modules:
            rs = [r for r in self._by_module.get(mod, []) if r.t_start_ms < we and r.t_end_ms > ws]
            if not rs:
                if (
                    not final
                    and mod in LATE_MODULES
                    and self._last_end.get(mod, 0) < we
                    and now_ms - we < self.late_horizon
                ):
                    availability[mod] = "pending"
                else:
                    availability[mod] = "missing"
                continue
            usable = [r for r in rs if r.status in (ResultStatus.ok, ResultStatus.partial)]
            availability[mod] = "available" if usable else rs[-1].status.value
            if any(r.finality == "provisional" for r in rs):
                provisional_inputs = True
            num: dict[str, list[tuple[float, float]]] = defaultdict(list)
            cat: dict[str, Counter[str]] = defaultdict(Counter)
            units: dict[str, str | None] = {}
            for r in usable:
                ov = min(r.t_end_ms, we) - max(r.t_start_ms, ws)
                w = max(ov, 1) / 1000.0
                for m in r.metrics:
                    if m.status not in (ResultStatus.ok, ResultStatus.partial) or m.value is None:
                        continue
                    key = f"{mod}.{m.name}"
                    units[key] = m.unit
                    v = m.value
                    if isinstance(v, bool):
                        num[key].append((float(v), w))
                    elif isinstance(v, (int, float)):
                        if math.isfinite(float(v)):
                            num[key].append((float(v), w))
                    elif isinstance(v, str):
                        cat[key][v] += 1
            for key, vals in num.items():
                tw = sum(w for _, w in vals)
                mean = sum(v * w for v, w in vals) / tw
                fused[key] = mean
                metrics.append(Metric(name=key, value=round(mean, 6), unit=units.get(key), method="overlap-weighted mean", kind="derived", group=mod))
            for key, c in cat.items():
                label, _ = c.most_common(1)[0]
                fused[key] = label
                metrics.append(Metric(name=key, value=label, method="mode", kind="derived", group=mod))
        alignment = window_alignment(fused)
        pending = any(v == "pending" for v in availability.values())
        finality = "final" if final or not (provisional_inputs or pending) and now_ms - we > self.late_horizon else "provisional"
        n_avail = sum(1 for v in availability.values() if v == "available")
        res = AnalysisResult(
            session_id=self.session_id,
            module_id=self.module_id,
            model_id="late_fusion",
            model_version=self.version,
            t_start_ms=ws,
            t_end_ms=we,
            metrics=metrics,
            status=ResultStatus.ok if n_avail else ResultStatus.insufficient_data,
            finality=finality,
            payload={"replace_key": ws, "availability": availability, "alignment": alignment},
        )
        self.computed[ws] = res
        return res
