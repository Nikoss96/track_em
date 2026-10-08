import { useEffect, useState } from "react";
import { api } from "../api";
import { MODULE_TITLES } from "../types";

type Field = { path: string; label: string; type: "number" | "text" | "bool" | "select"; options?: string[] };

const FIELDS: { title: string; fields: Field[] }[] = [
  { title: "Захват", fields: [
    { path: "capture.video_fps", label: "FPS видео", type: "number" },
    { path: "capture.audio_chunk_ms", label: "Аудиочанк, мс (100–250)", type: "number" },
    { path: "capture.preview_fps", label: "Превью, кадров/с", type: "number" },
    { path: "capture.silence_dbfs", label: "Порог тишины, dBFS", type: "number" },
  ] },
  { title: "Производительность", fields: [
    { path: "profile", label: "Профиль", type: "select", options: ["auto", "gpu", "cpu"] },
    { path: "inference.max_parallel", label: "Параллельных тяжёлых задач", type: "number" },
    { path: "inference.threads", label: "Потоков inference", type: "number" },
  ] },
  { title: "Мимика", fields: [
    { path: "modules.face.enabled", label: "Включено", type: "bool" },
    { path: "modules.face.analysis_fps", label: "Частота анализа, к/с", type: "number" },
    { path: "modules.face.window_ms", label: "Окно, мс", type: "number" },
  ] },
  { title: "Интонация", fields: [
    { path: "modules.prosody.enabled", label: "Включено", type: "bool" },
    { path: "modules.prosody.window_ms", label: "Окно, мс", type: "number" },
    { path: "modules.prosody.hop_ms", label: "Шаг, мс", type: "number" },
    { path: "modules.prosody.context_window_ms", label: "Контекстное окно, мс", type: "number" },
    { path: "modules.prosody.f0_min_hz", label: "F0 min, Гц", type: "number" },
    { path: "modules.prosody.f0_max_hz", label: "F0 max, Гц", type: "number" },
    { path: "modules.prosody.pause_min_ms", label: "Мин. пауза, мс", type: "number" },
    { path: "modules.prosody.egemaps", label: "eGeMAPS", type: "bool" },
  ] },
  { title: "Речь (ASR)", fields: [
    { path: "modules.speech.enabled", label: "Включено", type: "bool" },
    { path: "modules.speech.model", label: "Модель Whisper", type: "select", options: ["small", "medium", "large-v3-turbo", "large-v3"] },
    { path: "modules.speech.device", label: "Устройство", type: "select", options: ["auto", "cuda", "cpu"] },
    { path: "modules.speech.compute_type", label: "Точность", type: "select", options: ["auto", "float16", "int8_float16", "int8"] },
    { path: "modules.speech.beam_size", label: "Beam size", type: "number" },
    { path: "modules.speech.vad_threshold", label: "Порог VAD", type: "number" },
    { path: "modules.speech.min_silence_ms", label: "Тишина конца фразы, мс", type: "number" },
    { path: "modules.speech.max_segment_ms", label: "Макс. сегмент, мс", type: "number" },
    { path: "modules.speech.partial_interval_ms", label: "Интервал partial, мс (0 — выкл.)", type: "number" },
  ] },
  { title: "Голосовая экспрессия", fields: [
    { path: "modules.voice_emotion.enabled", label: "Включено", type: "bool" },
    { path: "modules.voice_emotion.model", label: "Модель", type: "select", options: ["emotion2vec/emotion2vec_plus_base", "emotion2vec/emotion2vec_plus_large", "emotion2vec/emotion2vec_plus_seed"] },
    { path: "modules.voice_emotion.device", label: "Устройство", type: "select", options: ["cuda", "cpu"] },
    { path: "modules.voice_emotion.window_ms", label: "Окно, мс", type: "number" },
    { path: "modules.voice_emotion.hop_ms", label: "Шаг, мс", type: "number" },
    { path: "modules.voice_emotion.min_speech_ms", label: "Мин. речь, мс", type: "number" },
  ] },
  { title: "Текст / транзактный анализ (LLM)", fields: [
    { path: "modules.text.enabled", label: "Семантика текста", type: "bool" },
    { path: "modules.transactional.enabled", label: "Транзактный анализ", type: "bool" },
    { path: "llm.backend", label: "Бэкенд LLM", type: "select", options: ["ollama", "openai"] },
    { path: "llm.allow_external", label: "Разрешить отправку текста во внешний сервис (OpenAI)", type: "bool" },
    { path: "llm.openai_model", label: "Модель OpenAI", type: "select", options: ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5.4-mini", "gpt-5.6-luna", "gpt-6-luna", "gpt-4.1"] },
    { path: "llm.openai_reasoning_effort", label: "Reasoning effort (только gpt-5*)", type: "select", options: ["", "minimal", "low", "medium"] },
    { path: "llm.session_budget_usd", label: "Лимит стоимости на сессию, $ (0 — без лимита)", type: "number" },
    { path: "llm.model", label: "Модель Ollama", type: "text" },
    { path: "llm.timeout_s", label: "Таймаут, с", type: "number" },
  ] },
  { title: "Fusion и хранение", fields: [
    { path: "fusion.window_ms", label: "Окно fusion, мс", type: "number" },
    { path: "fusion.step_ms", label: "Шаг fusion, мс", type: "number" },
    { path: "storage.save_media", label: "Сохранять медиазапись Live", type: "bool" },
    { path: "storage.parquet", label: "Parquet временных рядов", type: "bool" },
    { path: "storage.retention_days", label: "Хранить сессии, дней (0 — бессрочно)", type: "number" },
    { path: "scoring.composite.enabled", label: "Composite по умолчанию (экспериментальный)", type: "bool" },
  ] },
];

const get = (o: any, p: string) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
function setPath(o: any, p: string, v: any) {
  const ks = p.split(".");
  let cur = o;
  ks.slice(0, -1).forEach((k) => { cur[k] ??= {}; cur = cur[k]; });
  cur[ks[ks.length - 1]] = v;
}

function OpenAIKey() {
  const [st, setSt] = useState<any>(null);
  const [key, setKey] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { fetch("/api/secrets").then((r) => r.json()).then(setSt); }, []);
  async function save(value: string | null) {
    const r = await fetch("/api/secrets", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ openai_api_key: value }) });
    setSt(await r.json());
    setKey("");
    setMsg(value ? "Ключ сохранён в configs/secrets.yaml (не возвращается через API)." : "Ключ удалён.");
  }
  return (
    <section className="card">
      <h3>OpenAI API-ключ</h3>
      <div className="hint">
        Статус: {st?.openai_api_key_set ? `задан (${st.openai_api_key_source === "env" ? "переменная OPENAI_API_KEY" : "configs/secrets.yaml"})` : "не задан"}.
        В OpenAI отправляется только текст (фрагменты эталона, транскрипты, вопрос), не аудио и не видео.
        Для исследований с участниками это должно быть отражено в информированном согласии.
      </div>
      <div className="row">
        <input type="password" placeholder="sk-..." value={key} onChange={(e) => setKey(e.target.value)} style={{ maxWidth: 420 }} />
        <button disabled={!key.startsWith("sk-")} onClick={() => save(key)}>Сохранить ключ</button>
        <button disabled={st?.openai_api_key_source !== "file"} onClick={() => save(null)}>Удалить</button>
        {msg && <span className="hint">{msg}</span>}
      </div>
    </section>
  );
}

export default function Settings() {
  const [cfg, setCfg] = useState<any>(null);
  const [patch, setPatch] = useState<any>({});
  const [models, setModels] = useState<any[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  const load = () => { api.config().then(setCfg); api.modules().then((m) => setModels(m.models)); };
  useEffect(load, []);
  if (!cfg) return <div className="hint">загрузка…</div>;

  const val = (p: string) => { const v = get(patch, p); return v === undefined ? get(cfg, p) : v; };
  const change = (f: Field, raw: any) => {
    const v = f.type === "number" ? (raw === "" ? null : Number(raw)) : f.type === "select" && raw === "" ? null : raw;
    const n = structuredClone(patch);
    setPath(n, f.path, v);
    setPatch(n);
  };
  async function save() {
    try {
      const c = await api.putConfig(patch);
      setCfg(c);
      setPatch({});
      setMsg("Сохранено в configs/user.yaml. Применяется к новым сессиям.");
    } catch (e) {
      setMsg(String(e));
    }
  }

  return (
    <div>
      <div className="grid3">
        {FIELDS.map((g) => (
          <section key={g.title} className="card">
            <h3>{g.title}</h3>
            {g.fields.map((f) => (
              <label key={f.path} className={f.type === "bool" ? "row" : ""}>
                {f.type === "bool" ? (
                  <><input type="checkbox" checked={!!val(f.path)} onChange={(e) => change(f, e.target.checked)} /> {f.label}</>
                ) : (
                  <>{f.label}
                    {f.type === "select" ? (
                      <select value={val(f.path) ?? ""} onChange={(e) => change(f, e.target.value)}>
                        {[...new Set([...(f.options ?? []), String(val(f.path) ?? "")])].filter((o) => o !== "" || f.options?.includes("")).map((o) => <option key={o} value={o}>{o || "—"}</option>)}
                      </select>
                    ) : (
                      <input type={f.type} value={val(f.path) ?? ""} onChange={(e) => change(f, e.target.value)} />
                    )}
                  </>
                )}
              </label>
            ))}
          </section>
        ))}
      </div>
      <div className="buttons">
        <button className="primary" disabled={!Object.keys(patch).length} onClick={save}>Сохранить</button>
        <button disabled={!Object.keys(patch).length} onClick={() => setPatch({})}>Отменить</button>
        <span className="hint">Активный профиль: {cfg.profile_resolved}. {msg}</span>
      </div>
      <OpenAIKey />
      <section className="card">
        <h3>Реестр моделей</h3>
        <table>
          <thead><tr><th>Модель</th><th>Версия</th><th>Backend</th><th>Лицензия</th><th>Языки</th><th>Зарегистрирована</th></tr></thead>
          <tbody>{models.map((m) => <tr key={m.model_id + m.version}><td>{m.model_id}</td><td className="small">{String(m.version).slice(0, 24)}</td><td className="small">{m.backend}</td><td className="small">{m.license}</td><td className="small">{m.languages}</td><td className="small">{m.registered_at}</td></tr>)}</tbody>
        </table>
        <div className="hint">Модули: {Object.keys(MODULE_TITLES).join(", ")}. Отключение модуля не влияет на работу остальных.</div>
      </section>
    </div>
  );
}
