import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Brush } from "recharts";
import { api } from "../api";
import type { AnalysisResult } from "../types";
import { MODULE_TITLES, mstr, mval } from "../types";

const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777"];
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function valStr(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "string") return v;
  return JSON.stringify(v).slice(0, 80);
}

export default function Results({ sessionId, onSelect }: { sessionId: string | null; onSelect: (id: string) => void }) {
  const [sessions, setSessions] = useState<any[]>([]);
  const [data, setData] = useState<any | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<string[]>(["face.expression_intensity", "prosody.f0_sd_st", "text.valence"]);
  const [filt, setFilt] = useState({ module: "", from: "", to: "", status: "" });
  const [page, setPage] = useState(0);

  useEffect(() => { api.sessions().then(setSessions).catch(() => {}); }, [sessionId]);
  useEffect(() => {
    if (!sessionId) return;
    setData(null);
    api.results(sessionId).then(setData).catch((e) => setErr(String(e)));
  }, [sessionId]);

  const results: AnalysisResult[] = data?.results ?? [];
  const summary = data?.summary ?? {};
  const fusion = useMemo(() => results.filter((r) => r.module_id === "fusion").sort((a, b) => a.t_start_ms - b.t_start_ms), [results]);
  const fusionNames = useMemo(() => {
    const s = new Set<string>();
    fusion.forEach((r) => r.metrics.forEach((m) => typeof m.value === "number" && s.add(m.name)));
    return [...s].sort();
  }, [fusion]);
  const timeline = useMemo(() => fusion.map((r) => {
    const row: any = { t: (r.t_start_ms + r.t_end_ms) / 2000 };
    sel.forEach((n) => { row[n] = mval(r, n); });
    return row;
  }), [fusion, sel]);
  const prosody = useMemo(() => results.filter((r) => r.module_id === "prosody" && r.payload.scope === "instantaneous").map((r) => ({ t: r.t_end_ms / 1000, f0: mval(r, "f0_mean_hz"), rms: mval(r, "rms_dbfs") })), [results]);
  const face = useMemo(() => results.filter((r) => r.module_id === "face").map((r) => ({ t: r.t_end_ms / 1000, smile: mval(r, "smile"), brow: mval(r, "brow_raise"), mouth: mval(r, "mouth_open"), intensity: mval(r, "expression_intensity"), presence: mval(r, "face_presence") })), [results]);
  const speech = useMemo(() => results.filter((r) => r.module_id === "speech" && r.status === "ok" && !r.payload.summary).sort((a, b) => a.t_start_ms - b.t_start_ms), [results]);
  const segments = useMemo(() => speech.map((s) => {
    const ov = (r: AnalysisResult) => r.t_start_ms < s.t_end_ms && r.t_end_ms > s.t_start_ms;
    const voice = results.filter((r) => r.module_id === "voice_emotion" && r.status === "ok" && ov(r)).map((r) => mstr(r, "top_label"));
    const txt = results.filter((r) => r.module_id === "text" && r.status === "ok" && ov(r));
    return { s, voice, txt: txt[0] };
  }), [speech, results]);

  const filtered = useMemo(() => results.filter((r) => {
    if (filt.module && r.module_id !== filt.module) return false;
    if (filt.status && r.status !== filt.status) return false;
    if (filt.from && r.t_end_ms < +filt.from * 1000) return false;
    if (filt.to && r.t_start_ms > +filt.to * 1000) return false;
    return true;
  }), [results, filt]);
  const statusStats = useMemo(() => {
    const m: Record<string, Record<string, number>> = {};
    results.forEach((r) => { m[r.module_id] ??= {}; m[r.module_id][r.status] = (m[r.module_id][r.status] ?? 0) + 1; });
    return m;
  }, [results]);

  const scores = summary.scores;
  const cm = summary.crossmodal;
  const PAGE = 100;

  return (
    <div>
      <section className="card row">
        <label>Сессия
          <select value={sessionId ?? ""} onChange={(e) => onSelect(e.target.value)}>
            <option value="">—</option>
            {sessions.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.mode} · {s.source} · {s.state} · {s.created_at}</option>)}
          </select>
        </label>
        {sessionId && data?.session?.state === "completed" && (
          <span className="buttons">
            <a className="btn" href={api.exportUrl(sessionId, "json")}>Экспорт JSON</a>
            <a className="btn" href={api.exportUrl(sessionId, "csv")}>Экспорт CSV</a>
            <a className="btn" href={api.exportUrl(sessionId, "parquet")}>Parquet</a>
          </span>
        )}
        {data && data.session?.state !== "completed" && <span className="warn">сессия ещё не завершена — показаны промежуточные данные</span>}
      </section>
      {err && <div className="error">{err}</div>}
      {!data ? <div className="hint">Выберите сессию</div> : (
        <>
          <section className="card">
            <h3>Общая временная шкала (fusion, окна {5} с / шаг 1 с)</h3>
            <div className="chips">
              {fusionNames.map((n) => (
                <label key={n} className={sel.includes(n) ? "chip on" : "chip"}>
                  <input type="checkbox" checked={sel.includes(n)} onChange={(e) => setSel(e.target.checked ? [...sel, n] : sel.filter((x) => x !== n))} />{n}
                </label>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={timeline} syncId="res">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="t" domain={["dataMin", "dataMax"]} tickFormatter={fmt} />
                {sel.map((n, i) => <YAxis key={n} yAxisId={n} hide={i > 0} width={60} />)}
                <Tooltip labelFormatter={(v) => fmt(+v)} />
                <Legend />
                {sel.map((n, i) => <Line key={n} yAxisId={n} dataKey={n} stroke={PALETTE[i % PALETTE.length]} dot={false} isAnimationActive={false} connectNulls={false} />)}
                <Brush dataKey="t" height={20} tickFormatter={fmt} />
              </LineChart>
            </ResponsiveContainer>
            <div className="hint">Каждый ряд масштабирован по своей оси. Пустые участки — модальность отсутствует.</div>
          </section>

          <div className="grid2">
            <section className="card">
              <h3>Интонация</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={prosody} syncId="res">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="t" domain={["dataMin", "dataMax"]} tickFormatter={fmt} />
                  <YAxis yAxisId="f0" domain={[50, 400]} width={50} />
                  <YAxis yAxisId="rms" orientation="right" domain={[-70, 0]} width={40} />
                  <Tooltip labelFormatter={(v) => fmt(+v)} />
                  <Line yAxisId="f0" dataKey="f0" name="F0, Гц" stroke="#2563eb" dot={false} isAnimationActive={false} />
                  <Line yAxisId="rms" dataKey="rms" name="RMS, dBFS" stroke="#ea580c" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </section>
            <section className="card">
              <h3>Мимика</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={face} syncId="res">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="t" domain={["dataMin", "dataMax"]} tickFormatter={fmt} />
                  <YAxis domain={[0, 1]} width={40} />
                  <Tooltip labelFormatter={(v) => fmt(+v)} />
                  <Line dataKey="smile" name="улыбка" stroke="#16a34a" dot={false} isAnimationActive={false} />
                  <Line dataKey="intensity" name="интенсивность" stroke="#dc2626" dot={false} isAnimationActive={false} />
                  <Line dataKey="presence" name="лицо в кадре" stroke="#94a3b8" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </section>
          </div>

          <section className="card">
            <h3>Речевые сегменты</h3>
            <table>
              <thead><tr><th>Время</th><th>Распознано</th><th>WER</th><th>Голос (модель)</th><th>Окраска текста</th></tr></thead>
              <tbody>
                {segments.map(({ s, voice, txt }) => (
                  <tr key={s.payload.segment_id}>
                    <td>{fmt(s.t_start_ms / 1000)}–{fmt(s.t_end_ms / 1000)}</td>
                    <td>{s.payload.text}</td>
                    <td>{s.payload.alignment?.status === "ok" ? s.payload.alignment.wer.toFixed(2) : s.payload.alignment?.status ?? "—"}</td>
                    <td>{voice.join(", ") || "—"}</td>
                    <td>{txt ? `${mstr(txt, "sentiment")} (${mval(txt, "valence")?.toFixed(2)})` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {speech.some((s) => s.payload.alignment) && (
            <section className="card">
              <h3>Эталон vs произнесённое <span className="hint"><span className="op-sub">замена</span> <span className="op-del">пропуск</span> <span className="op-ins">вставка</span></span></h3>
              <div className="align">
                {speech.map((s) => s.payload.alignment?.status === "ok" ? (
                  <span key={s.payload.segment_id}>
                    {(s.payload.alignment.skipped_words ?? []).map((w: string, i: number) => <span key={`k${i}`} className="op-del">{w} </span>)}
                    {s.payload.alignment.ops.map((o: [string, string | null, string | null], i: number) => (
                      o[0] === "ok" ? <span key={i}>{o[1]} </span>
                        : o[0] === "sub" ? <span key={i} className="op-sub" title={`произнесено: ${o[2]}`}>{o[1]}→{o[2]} </span>
                          : o[0] === "del" ? <span key={i} className="op-del">{o[1]} </span>
                            : <span key={i} className="op-ins">+{o[2]} </span>
                    ))}
                    <span className="sep">‖ </span>
                  </span>
                ) : <span key={s.payload.segment_id} className="op-unal">[не сопоставлено: {s.payload.text}] </span>)}
              </div>
            </section>
          )}

          {scores && (
            <section className="card">
              <h3>ScoringEngine <span className="hint">v{scores.version}, нормализация {scores.normalization_version}</span></h3>
              <table>
                <thead><tr><th>Группа</th><th>Метрика</th><th>Raw</th><th>N</th><th>Normalized 0–100</th><th>Метод</th></tr></thead>
                <tbody>
                  {Object.entries(scores.groups).flatMap(([g, items]: any) => items.filter(() => g !== "crossmodal_consistency").map((it: any) => (
                    <tr key={g + it.name}>
                      <td>{g}</td><td>{it.name}</td>
                      <td className="small">{it.available ? valStr(it.raw) : <span className="hint">нет данных</span>}{it.sd != null ? ` ± ${it.sd}` : ""}</td>
                      <td>{it.n ?? ""}</td>
                      <td>{it.normalized ?? "—"}</td>
                      <td className="small">{it.method}{it.normalization ? ` | ${it.normalization.method} (${it.normalization.source})` : ""}</td>
                    </tr>
                  )))}
                </tbody>
              </table>
              <div className="experimental">
                <b>exploratory_composite_score:</b> {scores.exploratory_composite_score.enabled ? (scores.exploratory_composite_score.value ?? `null — ${scores.exploratory_composite_score.reason ?? ""}`) : "выключен"}
                <div className="small">{scores.exploratory_composite_score.label}</div>
              </div>
            </section>
          )}

          {cm && (
            <section className="card">
              <h3>Согласованность модальностей (описательно)</h3>
              <p className="hint">{cm.method_note} Расхождение модальностей не означает ошибку исполнения.</p>
              <table>
                <thead><tr><th>Пара</th><th>X</th><th>Y</th><th>ρ Спирмена</th><th>p</th><th>n окон</th></tr></thead>
                <tbody>{Object.entries(cm.spearman ?? {}).map(([k, v]: any) => <tr key={k}><td>{k}</td><td className="small">{v.x}</td><td className="small">{v.y}</td><td>{v.rho ?? "—"}</td><td>{v.p_value ?? "—"}</td><td>{v.n}</td></tr>)}</tbody>
              </table>
              <h4>Эмоционально значимые фрагменты (интенсивность текста ≥ {cm.emotional_fragments?.threshold}) vs остальные — Cohen's d</h4>
              <ul className="small">{Object.entries(cm.emotional_fragments?.effect_sizes ?? {}).map(([k, v]: any) => <li key={k}>{k}: d = {v.d ?? "—"} (n {v.n_a}/{v.n_b})</li>)}</ul>
              <h4>Паузы и границы предложений</h4>
              <div className="small">{cm.pauses_vs_sentences?.status === "ok" ? `пауз на границах: ${(cm.pauses_vs_sentences.pauses_at_boundaries_ratio * 100).toFixed(0)}%; границ с паузой: ${(cm.pauses_vs_sentences.boundaries_with_pause_ratio * 100).toFixed(0)}% (пауз ${cm.pauses_vs_sentences.pauses}, границ ${cm.pauses_vs_sentences.sentence_boundaries})` : "недостаточно данных"}</div>
            </section>
          )}

          <section className="card">
            <h3>Ошибки, пропуски, модели</h3>
            <table>
              <thead><tr><th>Модуль</th><th>Статусы результатов</th><th>Модель / версия</th><th>Состояние</th></tr></thead>
              <tbody>
                {(summary.modules ?? []).map((m: any) => (
                  <tr key={m.module_id}><td>{MODULE_TITLES[m.module_id] ?? m.module_id}</td>
                    <td className="small">{Object.entries(statusStats[m.module_id] ?? {}).map(([k, v]) => `${k}: ${v}`).join(", ")}</td>
                    <td className="small">{m.model_id} {m.model_version}</td><td className="small">{m.state}{m.detail ? ` — ${m.detail}` : ""}</td></tr>
                ))}
              </tbody>
            </table>
            {summary.llm_usage?.calls > 0 && (
              <div className="small">LLM ({summary.llm_usage.backend}): {summary.llm_usage.calls} вызовов (ошибок {summary.llm_usage.failed}), токены вх. {summary.llm_usage.input_tokens} (кэш {summary.llm_usage.cached_input_tokens}) / вых. {summary.llm_usage.output_tokens}{summary.llm_usage.reasoning_tokens ? ` (рассуждения ${summary.llm_usage.reasoning_tokens})` : ""}; стоимость ≈ ${summary.llm_usage.cost_usd.toFixed(4)}{summary.llm_usage.cost_complete ? "" : " (неполная: нет тарифа для модели)"}</div>
            )}
            {summary.capture && <div className="small">Аудио: разрывов {summary.capture.audio?.gaps}, потеряно {summary.capture.audio?.gap_ms} мс; видео: пустых кадров {summary.capture.video?.blank_frames}, повреждённых {summary.capture.video?.corrupted}</div>}
            {(summary.module_errors ?? []).length > 0 && <details><summary>Ошибки модулей ({summary.module_errors.length})</summary><ul className="small">{summary.module_errors.map((e: any, i: number) => <li key={i}>[{fmt(e.t_ms / 1000)}] {e.module_id}: {e.message}</li>)}</ul></details>}
          </section>

          <section className="card">
            <h3>Таблица результатов</h3>
            <div className="row">
              <select value={filt.module} onChange={(e) => { setFilt({ ...filt, module: e.target.value }); setPage(0); }}>
                <option value="">все модули</option>{Object.keys(statusStats).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
              <select value={filt.status} onChange={(e) => { setFilt({ ...filt, status: e.target.value }); setPage(0); }}>
                <option value="">все статусы</option>{["ok", "partial", "insufficient_data", "not_applicable", "unavailable", "error"].map((s) => <option key={s}>{s}</option>)}
              </select>
              <input placeholder="от, с" value={filt.from} onChange={(e) => { setFilt({ ...filt, from: e.target.value }); setPage(0); }} />
              <input placeholder="до, с" value={filt.to} onChange={(e) => { setFilt({ ...filt, to: e.target.value }); setPage(0); }} />
              <span className="hint">{filtered.length} записей</span>
              <button disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button>
              <button disabled={(page + 1) * PAGE >= filtered.length} onClick={() => setPage(page + 1)}>›</button>
            </div>
            <table>
              <thead><tr><th>Модуль</th><th>Интервал</th><th>Статус</th><th>Метрики</th></tr></thead>
              <tbody>
                {filtered.slice(page * PAGE, (page + 1) * PAGE).map((r, i) => (
                  <tr key={i}>
                    <td>{r.module_id}{r.finality === "provisional" ? " (prov.)" : ""}</td>
                    <td>{fmt(r.t_start_ms / 1000)}–{fmt(r.t_end_ms / 1000)}</td>
                    <td className={r.status === "ok" ? "" : "bad"}>{r.status}{r.error ? `: ${r.error}` : ""}</td>
                    <td className="small">{r.metrics.slice(0, 8).map((m) => `${m.name}=${valStr(m.value)}${m.unit ? " " + m.unit : ""}`).join("; ")}{r.metrics.length > 8 ? ` … (+${r.metrics.length - 8})` : ""}{r.quality.notes.length ? ` ⚠ ${r.quality.notes.join("; ")}` : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
