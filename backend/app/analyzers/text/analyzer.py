"""Module E: TextAnalyzer — семантика и эмоциональная окраска ТЕКСТА (не чтеца) локальной LLM."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analyzers.base import AnalyzerPlugin
from app.analyzers.llm_client import LLMError, grounded, llm_identity, make_llm
from app.analyzers.text.units import TextUnit, UnitCollector
from app.schemas.events import Event, EventType
from app.schemas.results import AnalysisResult, Metric, Quality, ResultStatus

M = ResultStatus
EMOTIONS = Literal["joy", "sadness", "anger", "fear", "surprise", "disgust", "tenderness", "anxiety", "admiration", "irony", "contempt", "neutral"]


class Construction(BaseModel):
    quote: str = Field(description="точная цитата из фрагмента")
    type: Literal["emotive_lexis", "negation", "exclamation", "comparison", "diminutive", "intensifier", "rhetorical_question", "interjection", "other"]
    effect: str = Field(description="коротко: как влияет на окраску", max_length=160)


class TextAnalysisOut(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    valence: float = Field(ge=-1, le=1)
    emotion_labels: list[EMOTIONS] = Field(max_length=4)
    emotional_intensity: float = Field(ge=0, le=1)
    constructions: list[Construction] = Field(default_factory=list, max_length=6)
    negations_considered: bool
    context: str = Field(max_length=300, description="как ближайший контекст влияет на оценку")
    source_type: Literal["narration", "character_speech", "mixed", "participant_answer"]
    explanation_short: str = Field(max_length=300)


SYSTEM = (
    "Ты — инструмент лингвистического анализа русского текста для научного исследования. "
    "Оцени эмоциональную окраску САМОГО ТЕКСТА (лексика, синтаксис, отрицания, контекст). "
    "Не делай выводов о чувствах, личности или состоянии читающего/говорящего человека. "
    "Учитывай отрицания («не рад» ≠ «рад») и ближайший контекст. "
    "Различай авторское повествование и прямую речь персонажей (реплики после тире или в кавычках). "
    "valence: −1 (крайне негативно) … 0 … +1 (крайне позитивно); emotional_intensity: 0 … 1. "
    "В constructions приводи ТОЛЬКО точные цитаты из анализируемого фрагмента. "
    "Ответ — строго JSON по схеме, все пояснения на русском, кратко."
)


def build_prompt(u: TextUnit) -> str:
    parts = []
    if u.source == "reference":
        parts.append("Режим: чтение художественного произведения. Анализируй фрагмент произведения (эталонный текст).")
    elif u.source == "participant_answer":
        parts.append("Режим: вопрос–ответ. Анализируй собственный ответ участника (распознанная речь, возможны ошибки ASR).")
        if u.question:
            parts.append(f"Вопрос: «{u.question}»")
    else:
        parts.append("Режим: чтение; эталона нет, анализируй распознанный текст произведения (возможны ошибки ASR).")
    if u.context_before:
        parts.append(f"Предшествующий контекст (не оценивать, только учитывать): «{u.context_before[-600:]}»")
    parts.append(f"АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ: «{u.text[:1500]}»")
    if u.context_after:
        parts.append(f"Следующее предложение (только для контекста): «{u.context_after[:300]}»")
    return "\n".join(parts)


class TextAnalyzer(AnalyzerPlugin):
    module_id = "text"
    version = "0.1.0"
    subscribes = (EventType.text_reference, EventType.transcript_final)
    queue_policy = "dedupe"
    queue_size = 64

    def dedupe_key(self, ev: Event) -> Any:
        return UnitCollector.dedupe_key(ev)

    async def initialize(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.llm = make_llm(self.ctx.config["llm"], self.ctx.shared.get("llm_usage"))
        await self.llm.check()
        self.model_id, self.model_version, reg = llm_identity(self.llm)
        self.ctx.shared["register_model"](**reg)
        self.units = UnitCollector(self.ctx.mode, bool(self.ctx.reference_text), self.ctx.question, int(config.get("min_chars", 12)))

    async def process(self, event: Event) -> list[AnalysisResult]:
        out = []
        for u in self.units.feed(event):
            out.append(await self.analyze(u))
        return out

    async def finalize(self, session_id: str) -> list[AnalysisResult]:
        return [await self.analyze(u) for u in self.units.flush()]

    async def analyze(self, u: TextUnit) -> AnalysisResult:
        payload: dict[str, Any] = {"text": u.text, "source": u.source, "key": u.key, "segment_ids": u.segment_ids,
                                   "note": "оценка текста; не является оценкой эмоционального состояния человека"}
        if len(u.text) < 3:
            return self.result(u.t_start_ms, u.t_end_ms, [], M.insufficient_data, payload=payload)
        try:
            async with (nullcontext() if self.llm.external else self.ctx.gate):  # внешний API не занимает локальный GPU
                res, meta = await self.llm.chat_json(SYSTEM, build_prompt(u), TextAnalysisOut, module=self.module_id)
        except LLMError as e:
            return self.result(u.t_start_ms, u.t_end_ms, [], M.error, error=str(e), payload=payload)
        q = Quality()
        good = [c for c in res.constructions if grounded(c.quote, u.text)]
        if len(good) < len(res.constructions):
            q.notes.append(f"отброшено цитат, отсутствующих в тексте: {len(res.constructions) - len(good)}")
        meth = f"LLM {self.llm.backend}/{self.llm.model}, structured JSON, temperature 0"
        metrics = [
            Metric(name="sentiment", value=res.sentiment, method=meth, kind="model_output", group="text"),
            Metric(name="valence", value=round(res.valence, 3), unit="[-1,1]", method=meth, kind="model_output", group="text"),
            Metric(name="emotional_intensity", value=round(res.emotional_intensity, 3), unit="[0,1]", method=meth, kind="model_output", group="text"),
            Metric(name="emotion_labels", value=list(res.emotion_labels), method=meth, kind="model_output", group="text"),
            Metric(name="source_type", value=res.source_type, method=meth, kind="model_output", group="text"),
        ]
        payload.update(
            {
                "constructions": [c.model_dump() for c in good],
                "negations_considered": res.negations_considered,
                "context": res.context,
                "explanation_short": res.explanation_short,
                "llm": meta,
            }
        )
        return self.result(u.t_start_ms, u.t_end_ms, metrics, M.ok, quality=q, latency_ms=meta.get("latency_ms", 0), payload=payload)
