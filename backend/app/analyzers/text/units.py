"""Сбор текстовых единиц для NLP-модулей (только завершённые фразы, без повторов).

Reading Mode + эталон: единица = фрагмент произведения из text.reference (с контекстом).
Reading без эталона: распознанный текст (source=transcript).
Q&A: собственный ответ участника, накопленный до min_chars, вместе с вопросом.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.events import Event, TextReference, Transcript


@dataclass
class TextUnit:
    text: str
    t_start_ms: int
    t_end_ms: int
    source: str  # reference | transcript | participant_answer
    context_before: str = ""
    context_after: str = ""
    question: str | None = None
    segment_ids: list[int] = field(default_factory=list)
    key: str = ""


class UnitCollector:
    def __init__(self, mode: str, has_reference: bool, question: str | None, min_chars: int) -> None:
        self.mode = mode
        self.has_reference = has_reference
        self.question = question
        self.min_chars = min_chars
        self.pending: list[Transcript] = []
        self.history: list[str] = []
        self.seen_idx: set[int] = set()

    @staticmethod
    def dedupe_key(ev: Event) -> Any:
        if isinstance(ev, TextReference):
            return ("ref", ev.sentence)
        if isinstance(ev, Transcript):
            return ("tr", ev.segment_id, ev.text)
        return id(ev)

    def feed(self, ev: Event) -> list[TextUnit]:
        if isinstance(ev, TextReference):
            if not (self.mode == "reading" and self.has_reference):
                return []
            # каждое предложение эталона анализируется один раз, даже если фрагменты сегментов перекрываются
            first = ev.sentence_from if ev.sentence_from >= 0 else ev.sentence_index
            sents = ev.sentences or [ev.sentence]
            new = [(first + k, s) for k, s in enumerate(sents) if (first + k) not in self.seen_idx]
            if not new:
                return []
            self.seen_idx.update(i for i, _ in new)
            prev = [s for k, s in enumerate(sents) if (first + k) < new[0][0]]
            before = " ".join([ev.context_before, *prev]).strip()
            text = " ".join(s for _, s in new)
            return [TextUnit(text, ev.t_ms, ev.t_end_ms, "reference", before, ev.context_after,
                             key=f"ref:{new[0][0]}-{new[-1][0]}", segment_ids=[ev.segment_id])]
        if isinstance(ev, Transcript):
            if self.mode == "reading" and self.has_reference:
                return []
            self.pending.append(ev)
            text = " ".join(p.text for p in self.pending).strip()
            if len(text) >= self.min_chars and text.rstrip().endswith((".", "!", "?", "…")):
                return [self._flush()]
            if len(text) >= 4 * self.min_chars:
                return [self._flush()]
        return []

    def _flush(self) -> TextUnit:
        ps, self.pending = self.pending, []
        text = " ".join(p.text for p in ps).strip()
        src = "participant_answer" if self.mode == "qa" else "transcript"
        u = TextUnit(text, ps[0].t_ms, ps[-1].t_end_ms, src, context_before=" ".join(self.history[-3:]), question=self.question if self.mode == "qa" else None,
                     segment_ids=[p.segment_id for p in ps], key=f"tr:{ps[0].segment_id}-{ps[-1].segment_id}")
        self.history.append(text)
        return u

    def flush(self) -> list[TextUnit]:
        if self.pending and " ".join(p.text for p in self.pending).strip():
            return [self._flush()]
        return []
