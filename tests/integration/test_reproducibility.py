"""Повторный анализ того же файла с той же конфигурацией даёт идентичные измерения."""

from tests.harness import Harness


def _snapshot(h: Harness):
    out = []
    for r in sorted(h.results, key=lambda r: (r.module_id, r.t_start_ms, r.t_end_ms, len(r.metrics))):
        out.append((r.module_id, r.t_start_ms, r.t_end_ms, r.status.value,
                    tuple((m.name, round(m.value, 6) if isinstance(m.value, float) else m.value) for m in r.metrics if not isinstance(m.value, list))))
    return out


async def test_same_file_same_results(fixtures):
    snaps = []
    for _ in range(2):
        h = Harness(["face", "prosody"])
        await h.run_file(fixtures / "reading_av.mp4", max_duration_s=12)
        snaps.append(_snapshot(h))
    assert len(snaps[0]) > 20
    assert snaps[0] == snaps[1]
