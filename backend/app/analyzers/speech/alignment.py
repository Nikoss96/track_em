"""Нормализация текста и последовательное выравнивание транскрипции с эталоном (Reading Mode).

Выравнивание — полуглобальное динамическое программирование по словам: гипотеза (фраза ASR)
выравнивается целиком, начало/конец участка эталона свободны внутри окна поиска вокруг
текущей позиции чтения. S/D/I и WER считаются только на выровненном завершённом фрагменте.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[a-zа-я0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?…])[\s»\"”)]*\s+")


def normalize_words(text: str) -> list[str]:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[-‐‑'’`]", "", t)  # дефис/апостроф внутри слова: «флёр-д'оранж» → «флердоранж»
    return _WORD_RE.findall(t)


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if para:
            parts.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return parts


@dataclass
class AlignOp:
    op: str  # ok | sub | del | ins
    ref: str | None
    hyp: str | None
    ref_index: int | None


@dataclass
class ReferenceText:
    text: str
    sentences: list[str] = field(init=False)
    words: list[str] = field(init=False)
    word_sentence: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.sentences = split_sentences(self.text)
        self.words, self.word_sentence = [], []
        for i, s in enumerate(self.sentences):
            ws = normalize_words(s)
            self.words.extend(ws)
            self.word_sentence.extend([i] * len(ws))


def semiglobal(hyp: list[str], ref: list[str]) -> tuple[int, int, int, list[AlignOp]]:
    """Возвращает (cost, ref_start, ref_end, ops); начало/конец ref свободны."""
    n, m = len(hyp), len(ref)
    INF = 10**9
    D = [[0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diag, 2 up(ins), 3 left(del)
    for i in range(1, n + 1):
        D[i][0] = i
        B[i][0] = 2
        hi = hyp[i - 1]
        Di, Dp, Bi = D[i], D[i - 1], B[i]
        for j in range(1, m + 1):
            best, b = Dp[j - 1] + (0 if hi == ref[j - 1] else 1), 1
            v = Dp[j] + 1
            if v < best:
                best, b = v, 2
            v = Di[j - 1] + 1
            if v < best:
                best, b = v, 3
            Di[j], Bi[j] = best, b
    end, cost = 0, INF
    for j in range(m + 1):  # при равной стоимости — более дальний конец (замена вместо вставки)
        if D[n][j] <= cost:
            cost, end = D[n][j], j
    ops: list[AlignOp] = []
    i, j = n, end
    while i > 0:
        b = B[i][j]
        if b == 1:
            ops.append(AlignOp("ok" if hyp[i - 1] == ref[j - 1] else "sub", ref[j - 1], hyp[i - 1], j - 1))
            i, j = i - 1, j - 1
        elif b == 2:
            ops.append(AlignOp("ins", None, hyp[i - 1], None))
            i -= 1
        else:
            ops.append(AlignOp("del", ref[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return cost, j, end, ops


class ReadingTracker:
    """Отслеживает позицию чтения в эталоне и накапливает S/D/I по завершённым фрагментам."""

    def __init__(self, reference: str, back: int = 10, max_skip: int = 8, max_cost_ratio: float = 0.6) -> None:
        self.ref = ReferenceText(reference)
        self.pointer = 0
        self.back = back
        self.max_skip = max_skip
        self.max_cost_ratio = max_cost_ratio
        self.totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0, "hits": 0, "segments": 0}

    @property
    def finished(self) -> bool:
        return self.pointer >= len(self.ref.words)

    def align(self, hyp_text: str) -> dict[str, Any]:
        hyp = normalize_words(hyp_text)
        if not hyp:
            return {"status": "insufficient_data", "reason": "пустая гипотеза"}
        if not self.ref.words:
            return {"status": "insufficient_data", "reason": "пустой эталон"}
        w0 = max(0, self.pointer - self.back)
        w1 = min(len(self.ref.words), self.pointer + max(50, 3 * len(hyp) + 20))
        window = self.ref.words[w0:w1]
        cost, s, e, ops = semiglobal(hyp, window)
        s, e = s + w0, e + w0
        for o in ops:
            if o.ref_index is not None:
                o.ref_index += w0
        if cost > self.max_cost_ratio * len(hyp) or e <= s:
            return {"status": "unaligned", "cost": cost, "hyp_words": len(hyp), "reason": "фраза не сопоставляется с эталоном рядом с текущей позицией"}
        skipped: list[str] = []
        jump = False
        if s > self.pointer:
            gap = s - self.pointer
            if gap <= self.max_skip:
                skipped = self.ref.words[self.pointer : s]
            else:
                jump = True
        S = sum(o.op == "sub" for o in ops)
        D = sum(o.op == "del" for o in ops) + len(skipped)
        I = sum(o.op == "ins" for o in ops)
        H = sum(o.op == "ok" for o in ops)
        N = (e - s) + len(skipped)
        self.pointer = max(self.pointer, e)
        for k, v in (("substitutions", S), ("deletions", D), ("insertions", I), ("ref_words", N), ("hits", H), ("segments", 1)):
            self.totals[k] += v
        sent_from = self.ref.word_sentence[s - len(skipped)] if N else None
        sent_to = self.ref.word_sentence[e - 1] if e > 0 else None
        return {
            "status": "ok",
            "ref_start": s - len(skipped),
            "ref_end": e,
            "substitutions": S,
            "deletions": D,
            "insertions": I,
            "hits": H,
            "ref_words": N,
            "wer": round((S + D + I) / N, 4) if N else None,
            "skipped_words": skipped,
            "jump": jump,
            "sentence_from": sent_from,
            "sentence_to": sent_to,
            "ops": [[o.op, o.ref, o.hyp] for o in ops],
            "cumulative": self.cumulative(),
        }

    def cumulative(self) -> dict[str, Any]:
        t = dict(self.totals)
        t["wer"] = round((t["substitutions"] + t["deletions"] + t["insertions"]) / t["ref_words"], 4) if t["ref_words"] else None
        t["progress"] = round(self.pointer / len(self.ref.words), 4) if self.ref.words else None
        return t

    def sentence(self, i: int) -> str:
        return self.ref.sentences[i] if 0 <= i < len(self.ref.sentences) else ""
