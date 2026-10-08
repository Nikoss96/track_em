from app.analyzers.speech.alignment import ReadingTracker, normalize_words, semiglobal, split_sentences

REF = "Мама мыла раму. Папа читал газету! Кот спал на окне? Ёжик шёл домой."


def test_normalize():
    assert normalize_words("Ёжик, флёр-д'оранж — «Да!»") == ["ежик", "флердоранж", "да"]


def test_sentences():
    assert split_sentences(REF) == ["Мама мыла раму.", "Папа читал газету!", "Кот спал на окне?", "Ёжик шёл домой."]


def test_semiglobal_counts():
    cost, s, e, ops = semiglobal(["папа", "читал", "газеты"], ["мама", "мыла", "раму", "папа", "читал", "газету", "кот"])
    assert (cost, s, e) == (1, 3, 6)
    assert [o.op for o in ops] == ["ok", "ok", "sub"]


def test_sequential_tracking_and_wer_on_read_part_only():
    tr = ReadingTracker(REF)
    a = tr.align("мама мыла раму")
    assert a["status"] == "ok" and a["wer"] == 0 and a["sentence_to"] == 0
    b = tr.align("папа читал свою газету")  # вставка
    assert (b["insertions"], b["substitutions"], b["deletions"], b["ref_words"]) == (1, 0, 0, 3)
    c = tr.align("спал на окне")  # пропущено слово «кот» → deletion
    assert c["deletions"] == 1 and c["skipped_words"] == ["кот"]
    cum = tr.cumulative()
    # WER только по прочитанной части (10 слов из 12), а не по всему эталону
    assert cum["ref_words"] == 10 and cum["wer"] == round(2 / 10, 4)
    assert cum["progress"] < 1


def test_unrelated_phrase_not_aligned():
    tr = ReadingTracker(REF)
    r = tr.align("совершенно другой текст про погоду и цены")
    assert r["status"] == "unaligned"
    assert tr.pointer == 0 and tr.totals["ref_words"] == 0
