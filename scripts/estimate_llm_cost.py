"""Оценка стоимости LLM-аналитики через OpenAI API (до запуска сессий).

    .venv\\Scripts\\python.exe scripts\\estimate_llm_cost.py [--reference text.txt] [--wpm 130] [--hours 1]

Токены считаются токенизатором o200k_base (tiktoken) по РЕАЛЬНЫМ промптам приложения:
system + user (фрагмент + контекст) + JSON Schema ответа (передаётся в response_format и тарифицируется как вход),
выход — по типичному ответу. Частота вызовов:
- Reading: одна текстовая единица на фразу ASR ≈ предложение эталона; транзактный вызов — только для фрагментов
  с прямой речью (доля берётся из эталона);
- Q&A: единица ≈ 25 слов речи участника; транзактный вызов на каждую единицу.
Фактическая стоимость сессии считается по usage из ответов API и показывается в UI («Результаты»).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import tiktoken  # noqa: E402

from app.analyzers.llm_client import strict_schema  # noqa: E402
from app.analyzers.speech.alignment import split_sentences  # noqa: E402
from app.analyzers.text.analyzer import SYSTEM as TEXT_SYSTEM  # noqa: E402
from app.analyzers.text.analyzer import TextAnalysisOut, build_prompt  # noqa: E402
from app.analyzers.text.units import TextUnit  # noqa: E402
from app.analyzers.transactional.analyzer import SYSTEM as TA_SYSTEM  # noqa: E402
from app.analyzers.transactional.analyzer import TransactionalAnalyzer, TransactionalOut, has_direct_speech  # noqa: E402
from app.config import load_config  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")
MSG_OVERHEAD = 8  # служебные токены сообщений chat-формата (приблизительно)

TEXT_OUT_SAMPLE = {
    "sentiment": "mixed", "valence": 0.2, "emotion_labels": ["joy", "surprise"], "emotional_intensity": 0.6,
    "constructions": [
        {"quote": "Голубчик мой!", "type": "exclamation", "effect": "передаёт тёплую радость встречи"},
        {"quote": "Сколько зим, сколько лет!", "type": "emotive_lexis", "effect": "усиливает эмоциональность приветствия"},
    ],
    "negations_considered": True,
    "context": "Фрагмент описывает неожиданную встречу старых друзей на вокзале.",
    "source_type": "character_speech",
    "explanation_short": "Восклицания и ласковое обращение создают позитивную окраску; контекст бытовой, без негативной лексики.",
}
TA_OUT_SAMPLE = {
    "status": "ok",
    "replicas": [
        {"speaker": "толстый", "quote": "Порфирий! Ты ли это? Голубчик мой!", "label": "child_free", "uncertainty": "medium",
         "rationale": "Спонтанная эмоциональная реакция, восклицания, ласковое обращение.",
         "evidence": [{"marker": "spontaneous_emotion", "quote": "Голубчик мой!"}]},
        {"speaker": "тонкий", "quote": "Батюшки! Миша! Друг детства!", "label": "child_free", "uncertainty": "medium",
         "rationale": "Эмоциональное удивление без оценочных предписаний.",
         "evidence": [{"marker": "spontaneous_emotion", "quote": "Батюшки!"}]},
    ],
    "note": "",
}


def ntok(s: str) -> int:
    return len(ENC.encode(s))


def schema_tokens(model: type) -> int:
    return ntok(json.dumps(strict_schema(model.model_json_schema()), ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(ROOT / "tests/fixtures/reading_reference.txt"))
    ap.add_argument("--wpm", type=float, default=130, help="темп чтения/речи, слов в минуту")
    ap.add_argument("--hours", type=float, default=1.0, help="часы записи")
    ap.add_argument("--qa-speech-share", type=float, default=0.6, help="доля времени Q&A, когда участник говорит")
    ap.add_argument("--words-per-phrase", type=float, default=12, help="слов в фразе ASR при чтении (между паузами)")
    ap.add_argument("--direct-share", type=float, default=None, help="доля фраз с прямой речью (по умолчанию — из эталона)")
    ap.add_argument("--retry-overhead", type=float, default=1.05, help="запас на повторы/невалидный JSON")
    a = ap.parse_args()

    ref = Path(a.reference).read_text(encoding="utf-8")
    sents = split_sentences(ref)
    words_per_sent = sum(len(s.split()) for s in sents) / len(sents)

    # Reading: средние токены по реальным единицам эталона
    t_in, ta_in, n_ta = [], [], 0
    ta = TransactionalAnalyzer.__new__(TransactionalAnalyzer)
    for i, s in enumerate(sents):
        u = TextUnit(s, 0, 0, "reference", context_before=" ".join(sents[max(0, i - 2):i]), context_after=sents[i + 1] if i + 1 < len(sents) else "")
        t_in.append(ntok(TEXT_SYSTEM) + ntok(build_prompt(u)) + 2 * MSG_OVERHEAD)
        if has_direct_speech(s):
            n_ta += 1
            ta_in.append(ntok(TA_SYSTEM) + ntok(ta._prompt(u)) + 2 * MSG_OVERHEAD)
    t_schema, ta_schema = schema_tokens(TextAnalysisOut), schema_tokens(TransactionalOut)
    t_out = ntok(json.dumps(TEXT_OUT_SAMPLE, ensure_ascii=False))
    ta_out = ntok(json.dumps(TA_OUT_SAMPLE, ensure_ascii=False))
    text_call = (sum(t_in) / len(t_in) + t_schema, t_out)
    ta_call = ((sum(ta_in) / len(ta_in) if ta_in else 400) + ta_schema, ta_out)
    direct_share = a.direct_share if a.direct_share is not None else n_ta / len(sents)

    # единица Reading = фраза ASR (до паузы ≥ 600 мс) с новыми предложениями; короткие реплики объединяются в фразе
    units_per_h_reading = a.wpm * 60 / max(words_per_sent, a.words_per_phrase)
    units_per_h_qa = a.wpm * 60 * a.qa_speech_share / 25
    # Q&A: длиннее единица (≈25 слов) и есть вопрос; оценка входа — как reading + 30 токенов вопроса
    qa_text_call = (text_call[0] + 40, t_out)
    qa_ta_call = (ta_call[0] + 40, ta_out)

    scenarios = {
        "reading": [(units_per_h_reading, text_call), (units_per_h_reading * direct_share, ta_call)],
        "qa": [(units_per_h_qa, qa_text_call), (units_per_h_qa, qa_ta_call)],
    }
    print(f"Эталон: {len(sents)} предложений, {words_per_sent:.1f} слов/предл., прямая речь в {direct_share:.0%} предложений")
    print(f"Текстовый вызов: вход ≈ {text_call[0]:.0f} ток. (из них схема {t_schema}), выход ≈ {t_out}")
    print(f"Транзактный вызов: вход ≈ {ta_call[0]:.0f} ток. (схема {ta_schema}), выход ≈ {ta_out}")
    for name, calls in scenarios.items():
        n = sum(c for c, _ in calls) * a.hours
        tin = sum(c * io[0] for c, io in calls) * a.hours * a.retry_overhead
        tout = sum(c * io[1] for c, io in calls) * a.hours * a.retry_overhead
        print(f"\n== {name}: {a.hours:g} ч записи, {a.wpm:.0f} слов/мин → ≈ {n:.0f} вызовов, {tin / 1e3:.0f}K вх. + {tout / 1e3:.0f}K вых. токенов")
        print(f"{'модель':16} {'$ за сценарий':>14} {'$ / 10 ч':>10}   примечание")
        for model, p in load_config("cpu")["llm"]["pricing"].items():
            # системный промпт и схема повторяются → часть входа попадает в prompt cache (≥1024 ток. префикса);
            # наш префикс < 1024 ток., поэтому кэш не учитываем (консервативно)
            cost = (tin * p["input"] + tout * p["output"]) / 1e6
            note = "reasoning: + токены рассуждений (см. ниже)" if model.startswith(("gpt-5", "gpt-6")) else ""
            print(f"{model:16} {cost:14.4f} {cost * 10 / a.hours:10.3f}   {note}")
        print("reasoning-модели: при effort=minimal обычно +0–200, при low +300–1000 вых. токенов на вызов (неточно).")


if __name__ == "__main__":
    main()
