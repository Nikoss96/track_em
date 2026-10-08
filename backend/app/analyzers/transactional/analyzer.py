"""Module F: TransactionalAnalyzer — наблюдаемые речевые паттерны по транзактному анализу Э. Берна.

Классификация коммуникативной позиции реплики по наблюдаемым маркерам; НЕ диагностика
эго-состояния. Reading Mode: только реплики персонажей (прямая речь) во фрагменте произведения —
позиция персонажа не приписывается чтецу; без прямой речи → not_applicable.
Q&A: ответ участника на вопрос; слишком короткий ответ → insufficient_data.
"""

from __future__ import annotations

import re
from collections import Counter
from contextlib import nullcontext
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analyzers.base import AnalyzerPlugin
from app.analyzers.llm_client import LLMError, grounded, llm_identity, make_llm
from app.analyzers.text.units import TextUnit, UnitCollector
from app.schemas.events import Event, EventType
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus

M = ResultStatus
Label = Literal["parent_critical", "parent_nurturing", "adult", "child_free", "child_adapted", "mixed", "undetermined"]
Marker = Literal["directivity", "condemnation", "patronizing", "care_support", "factual_argumentation", "rational_structure",
                 "emotional_protest", "spontaneous_emotion", "dependent_adaptive", "compliance_apology"]
DISCLAIMER = "Классификация наблюдаемых речевых паттернов (транзактный анализ); не диагностика психологического эго-состояния."

_DIRECT = re.compile(r"(^|\n)\s*[—–-]\s*\S|«[^»]{2,}»|\"[^\"]{2,}\"")


class Evidence(BaseModel):
    marker: Marker
    quote: str = Field(description="точная цитата из реплики")


class ReplicaOut(BaseModel):
    speaker: str = Field(max_length=60, description="персонаж/участник, если ясно из текста, иначе 'неизвестно'")
    quote: str = Field(description="реплика (точная цитата)")
    label: Label
    evidence: list[Evidence] = Field(default_factory=list, max_length=5)
    uncertainty: Literal["low", "medium", "high"]
    rationale: str = Field(max_length=300)


class TransactionalOut(BaseModel):
    status: Literal["ok", "insufficient_data", "not_applicable"]
    replicas: list[ReplicaOut] = Field(default_factory=list, max_length=8)
    note: str = Field(default="", max_length=300)


SYSTEM = (
    "Ты — инструмент анализа коммуникативных паттернов по транзактному анализу Эрика Берна для научного исследования. "
    "Классифицируй ТОЛЬКО наблюдаемые речевые маркеры реплик: "
    "parent_critical (директивность, осуждение, оценочные предписания), parent_nurturing (забота, покровительство, опека), "
    "adult (фактическая аргументация, рациональная структура, вопросы по существу), "
    "child_free (спонтанные эмоции, восторг, игра, протест без адаптации), child_adapted (зависимость, подстройка, оправдания, угодливость), "
    "mixed (несколько позиций сразу), undetermined (маркеров недостаточно). "
    "Это не психологическая диагностика: не делай выводов о личности или истинном состоянии. "
    "Каждую метку обосновывай точными цитатами (evidence.quote — дословно из реплики). "
    "Если реплик/контекста недостаточно — status=insufficient_data; если в тексте нет диалоговых реплик — status=not_applicable. "
    "Ответ — строго JSON по схеме, пояснения на русском, кратко."
)


def has_direct_speech(text: str) -> bool:
    return bool(_DIRECT.search(text))


class TransactionalAnalyzer(AnalyzerPlugin):
    module_id = "transactional"
    version = "0.1.0"
    subscribes = (EventType.text_reference, EventType.transcript_final)
    queue_policy = "dedupe"
    queue_size = 64

    def dedupe_key(self, ev: Event) -> Any:
        return UnitCollector.dedupe_key(ev)

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.min_words = int(config.get("min_words", 4))
        self.llm = make_llm(self.ctx.config["llm"], self.ctx.shared.get("llm_usage"))
        await self.llm.check()
        self.model_id, self.model_version, _ = llm_identity(self.llm)
        min_chars = int(self.ctx.config["modules"].get("text", {}).get("min_chars", 12))
        self.units = UnitCollector(self.ctx.mode, bool(self.ctx.reference_text), self.ctx.question, min_chars)

    async def process(self, event: Event) -> list[AnalysisResult]:
        return [await self.analyze(u) for u in self.units.feed(event)]

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        return [await self.analyze(u) for u in self.units.flush()]

    def _prompt(self, u: TextUnit) -> str:
        if u.source == "participant_answer":
            return "\n".join(filter(None, [
                "Режим: вопрос–ответ. Классифицируй ответ участника как одну реплику (speaker='участник').",
                f"Вопрос: «{u.question}»" if u.question else "Вопрос не задан.",
                f"Предыдущие фразы участника: «{u.context_before[-500:]}»" if u.context_before else "",
                f"ОТВЕТ: «{u.text[:1500]}»",
            ]))
        return "\n".join(filter(None, [
            "Режим: чтение художественного произведения. Классифицируй реплики ПЕРСОНАЖЕЙ (прямую речь) во фрагменте. "
            "Авторское повествование не классифицируй. Чтец произведения не является объектом анализа.",
            f"Предшествующий контекст: «{u.context_before[-600:]}»" if u.context_before else "",
            f"ФРАГМЕНТ: «{u.text[:1500]}»",
        ]))

    async def analyze(self, u: TextUnit) -> AnalysisResult:
        payload: dict[str, Any] = {"text": u.text, "source": u.source, "key": u.key, "disclaimer": DISCLAIMER,
                                   "subject": "participant" if u.source == "participant_answer" else "characters"}
        if u.source == "participant_answer":
            if len(u.text.split()) < self.min_words:
                return self.result(u.t_start_ms, u.t_end_ms, [], M.insufficient_data, payload={**payload, "reason": f"ответ короче {self.min_words} слов"})
        else:
            # в чтении классифицируются только персонажи; без прямой речи анализ неприменим
            text_ctx = u.text
            if not has_direct_speech(text_ctx):
                return self.result(u.t_start_ms, u.t_end_ms, [], M.not_applicable, payload={**payload, "reason": "во фрагменте нет прямой речи персонажей"})
        try:
            async with (nullcontext() if self.llm.external else self.ctx.gate):  # внешний API не занимает локальный GPU
                res, meta = await self.llm.chat_json(SYSTEM, self._prompt(u), TransactionalOut, module=self.module_id)
        except LLMError as e:
            return self.result(u.t_start_ms, u.t_end_ms, [], M.error, error=str(e), payload=payload)
        payload["llm"] = meta
        if res.status != "ok" or not res.replicas:
            st = M.not_applicable if res.status == "not_applicable" else M.insufficient_data
            return self.result(u.t_start_ms, u.t_end_ms, [], st, payload={**payload, "reason": res.note or res.status})
        q = Quality()
        replicas = []
        dropped = 0
        for rep in res.replicas:
            ev = [e for e in rep.evidence if grounded(e.quote, u.text)]
            dropped += len(rep.evidence) - len(ev)
            label = rep.label if (ev or rep.label == "undetermined") else "undetermined"
            replicas.append({**rep.model_dump(), "evidence": [e.model_dump() for e in ev], "label": label,
                             "label_downgraded": label != rep.label})
        if dropped:
            q.notes.append(f"отброшено обоснований без точной цитаты: {dropped}")
        labels = [r["label"] for r in replicas]
        c = Counter(labels)
        primary = c.most_common(1)[0][0] if len(c) == 1 or c.most_common(2)[0][1] > c.most_common(2)[1][1] else "mixed"
        meth = f"LLM {self.llm.backend}/{self.llm.model}; транзактный анализ (наблюдаемые маркеры)"
        metrics = [
            Metric(name="primary_label", value=primary, method=meth, kind="model_output", group="transactional"),
            Metric(name="replica_count", value=len(replicas), unit="count", method=meth, kind="model_output", group="transactional"),
        ]
        for lab, n in c.items():
            metrics.append(Metric(name=f"count_{lab}", value=n, unit="count", method=meth, kind="model_output", group="transactional"))
        unc = Counter(r["uncertainty"] for r in replicas)
        metrics.append(Metric(name="llm_reported_uncertainty", value=unc.most_common(1)[0][0], method="самоотчёт LLM (не калиброван)", kind="model_output", group="transactional"))
        payload["replicas"] = replicas
        return self.result(u.t_start_ms, u.t_end_ms, metrics, M.ok, quality=q, latency_ms=meta.get("latency_ms", 0), payload=payload)
