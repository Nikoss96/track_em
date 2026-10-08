import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";
import type { LiveState } from "../useLiveSession";
import { MODULE_TITLES, STATE_RU, mstr, mval } from "../types";

const WINDOW_S = 60;
const COLORS = { f0: "#2563eb", rms: "#ea580c", smile: "#16a34a", brow: "#9333ea", mouth: "#0891b2", intensity: "#dc2626" };

function fmt(t: number) {
  const s = Math.max(0, t);
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

export default function Live({ live }: { live: LiveState }) {
  const st = live.status;
  const cap = st?.capture;
  const tmax = live.tmax / 1000;
  const domain: [number, number] = [Math.max(0, tmax - WINDOW_S), Math.max(WINDOW_S, tmax)];
  const xs = { type: "number" as const, dataKey: "t", domain, tickFormatter: fmt, allowDataOverflow: true };
  const pros = live.prosody.map((p) => ({ t: p.t / 1000, f0: p.f0, rms: p.rms }));
  const face = live.face.map((f) => ({ t: f.t / 1000, smile: f.smile, brow: f.brow, mouth: f.mouth, intensity: f.intensity }));
  const pauses = live.pauses.filter((p) => p[1] / 1000 > domain[0]);
  const lastVoice = live.voice[live.voice.length - 1];
  const lastText = live.text.filter((r) => r.status === "ok").slice(-1)[0];
  const lastTr = live.transactional.slice(-1)[0];
  const level = cap?.audio.levels?.slice(-1)[0]?.[1] ?? -120;

  return (
    <div className="live">
      <div className="grid3">
        <section className="card">
          <h3>Превью {live.connected ? "" : <span className="warn">(нет соединения)</span>}</h3>
          {live.preview ? <img className="preview" src={`data:image/jpeg;base64,${live.preview}`} /> : <div className="placeholder">нет кадров</div>}
          <div className="kv">
            <span>Состояние</span><b>{st?.state ?? live.info?.state ?? "—"}</b>
            <span>Медиавремя</span><b>{fmt((st?.media_ms ?? 0) / 1000)}</b>
            <span>Видео</span><b className={cap?.video.no_signal ? "bad" : ""}>{cap ? `${cap.video.fps} к/с, ${cap.video.frames} кадров${cap.video.no_signal ? " — нет сигнала" : ""}` : "—"}</b>
            <span>Пустые кадры</span><b>{cap?.video.blank_frames ?? 0}</b>
            <span>Аудио</span><b className={cap?.audio.no_signal ? "bad" : ""}>{cap ? `${cap.audio.source_rate ?? "?"} Гц ×${cap.audio.source_channels ?? "?"}, ${cap.audio.seconds} с` : "—"}</b>
            <span>Потери аудио</span><b className={cap?.audio.gaps ? "bad" : ""}>{cap ? `${cap.audio.gaps} разрывов, ${cap.audio.gap_ms} мс` : "—"}</b>
          </div>
          <div className="meter"><div style={{ width: `${Math.max(0, Math.min(100, (level + 70) * 1.6))}%` }} /></div>
          <div className="hint">{level.toFixed(1)} dBFS {cap?.audio.silence ? " — тишина" : ""} {st?.vad?.in_speech ? " • речь" : ""}</div>
        </section>
        <section className="card span2">
          <h3>Транскрипция</h3>
          <div className="transcript">
            {live.finals.slice(-8).map((f) => <p key={f.segment_id}><span className="t">{fmt(f.t0 / 1000)}</span> {f.text}</p>)}
            {live.partial && <p className="partial">{live.partial.text} …</p>}
          </div>
          {live.reference && (<><h4>Текущий фрагмент эталона</h4><blockquote>{live.reference.sentence}</blockquote></>)}
          <div className="grid2 tight">
            <div>
              <h4>Голосовая экспрессия <span className="tag">выход модели, ru не валидирован</span></h4>
              {lastVoice ? (
                <div>
                  <b>{mstr(lastVoice, "top_label")}</b> <span className="hint">{fmt(lastVoice.t_start_ms / 1000)}–{fmt(lastVoice.t_end_ms / 1000)}</span>
                  <div className="bars">
                    {Object.entries(lastVoice.payload.scores ?? {}).sort((a: any, b: any) => b[1] - a[1]).slice(0, 5).map(([k, v]: any) => (
                      <div key={k} className="bar"><span>{k}</span><div style={{ width: `${v * 100}%` }} /><em>{v.toFixed(2)}</em></div>
                    ))}
                  </div>
                </div>
              ) : <div className="hint">нет данных</div>}
            </div>
            <div>
              <h4>Окраска текста <span className="tag">оценка текста, не чтеца</span></h4>
              {lastText ? (
                <div>
                  <b>{mstr(lastText, "sentiment")}</b>, валентность {mval(lastText, "valence")?.toFixed(2)}, интенсивность {mval(lastText, "emotional_intensity")?.toFixed(2)}
                  <div className="hint">{lastText.payload.explanation_short}</div>
                </div>
              ) : <div className="hint">нет данных</div>}
              {lastTr && (
                <div className="hint">Транзактный: {lastTr.status === "ok" ? mstr(lastTr, "primary_label") : lastTr.status} ({lastTr.payload.subject === "characters" ? "персонажи" : "участник"})</div>
              )}
            </div>
          </div>
        </section>
      </div>

      <section className="card">
        <h3>Интонация: F0 и энергия <span className="hint">(серым — паузы)</span></h3>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={pros} syncId="t" margin={{ left: 0, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis {...xs} />
            <YAxis yAxisId="f0" domain={[50, 400]} unit=" Гц" width={70} />
            <YAxis yAxisId="rms" orientation="right" domain={[-70, 0]} unit=" dB" width={60} />
            <Tooltip labelFormatter={(v) => fmt(+v)} />
            <Legend />
            {pauses.map(([a, b], i) => <ReferenceArea key={i} yAxisId="f0" x1={a / 1000} x2={b / 1000} fill="#94a3b8" fillOpacity={0.25} ifOverflow="hidden" />)}
            <Line yAxisId="f0" dataKey="f0" name="F0 (озвуч.)" stroke={COLORS.f0} dot={false} connectNulls={false} isAnimationActive={false} />
            <Line yAxisId="rms" dataKey="rms" name="RMS" stroke={COLORS.rms} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <h3>Мимика (MediaPipe blendshapes, 0–1)</h3>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={face} syncId="t" margin={{ left: 0, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis {...xs} />
            <YAxis domain={[0, 1]} width={70} />
            <Tooltip labelFormatter={(v) => fmt(+v)} />
            <Legend />
            <Line dataKey="smile" name="улыбка" stroke={COLORS.smile} dot={false} isAnimationActive={false} />
            <Line dataKey="brow" name="брови вверх" stroke={COLORS.brow} dot={false} isAnimationActive={false} />
            <Line dataKey="mouth" name="открытие рта" stroke={COLORS.mouth} dot={false} isAnimationActive={false} />
            <Line dataKey="intensity" name="интенсивность" stroke={COLORS.intensity} dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <h3>Модули</h3>
        <table>
          <thead><tr><th>Модуль</th><th>Состояние</th><th>Модель</th><th>Обработано</th><th>Очередь</th><th>Пропущено</th><th>Ошибки</th><th>Задержка, мс</th><th>Детали</th></tr></thead>
          <tbody>
            {(st?.modules ?? []).map((m) => (
              <tr key={m.module_id}>
                <td>{MODULE_TITLES[m.module_id] ?? m.module_id}</td>
                <td className={`state ${m.state}`}>{STATE_RU[m.state] ?? m.state}</td>
                <td className="small">{m.model_id ?? ""}</td>
                <td>{m.processed}</td><td>{m.queue_size}</td><td>{m.dropped}</td><td className={m.errors ? "bad" : ""}>{m.errors}</td>
                <td>{m.last_latency_ms?.toFixed(0) ?? "—"} / ср. {m.avg_latency_ms?.toFixed(0) ?? "—"}</td>
                <td className="small">{m.detail ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(st as any)?.llm_usage?.calls > 0 && (
          <div className="hint">LLM ({(st as any).llm_usage.backend}): {(st as any).llm_usage.calls} вызовов, {(st as any).llm_usage.input_tokens} вх. / {(st as any).llm_usage.output_tokens} вых. токенов, ≈ ${(st as any).llm_usage.cost_usd.toFixed(4)}{(st as any).llm_usage.budget_usd ? ` из лимита $${(st as any).llm_usage.budget_usd}` : ""}</div>
        )}
        {st?.gate && <div className="hint">Inference gate: {st.gate.active}/{st.gate.slots} активно, ожидают {st.gate.waiting}; VAD-сегментов {st.vad?.segments ?? 0}</div>}
        {live.errors.length > 0 && (
          <details open><summary className="bad">Ошибки модулей ({live.errors.length})</summary>
            <ul className="small">{live.errors.slice(-10).map((e, i) => <li key={i}>[{fmt(e.t_ms / 1000)}] {e.module_id}: {e.message}</li>)}</ul>
          </details>
        )}
      </section>
    </div>
  );
}
