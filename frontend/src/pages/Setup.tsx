import { useEffect, useState } from "react";
import { api } from "../api";
import { MODULE_TITLES } from "../types";

interface Props {
  sessionId: string | null;
  sessionState: string | null;
  onStarted: (id: string) => void;
  onReset: () => void;
}

export default function Setup({ sessionId, sessionState, onStarted, onReset }: Props) {
  const [mode, setMode] = useState<"reading" | "qa">("reading");
  const [source, setSource] = useState<"file" | "live">("file");
  const [devices, setDevices] = useState<{ monitors: any[]; audio_outputs: any[] }>({ monitors: [], audio_outputs: [] });
  const [modules, setModules] = useState<any[]>([]);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [monitor, setMonitor] = useState(0);
  const [region, setRegion] = useState({ left: 0, top: 0, width: 1280, height: 720, full: true });
  const [audioDevice, setAudioDevice] = useState<string>("");
  const [fps, setFps] = useState(30);
  const [file, setFile] = useState<{ path: string; probe?: any } | null>(null);
  const [filePath, setFilePath] = useState("");
  const [realtime, setRealtime] = useState(true);
  const [refText, setRefText] = useState("");
  const [question, setQuestion] = useState("");
  const [profile, setProfile] = useState("auto");
  const [saveMedia, setSaveMedia] = useState(false);
  const [composite, setComposite] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.devices().then(setDevices).catch((e) => setErr(String(e)));
    api.modules().then((m) => {
      setModules(m.modules);
      setEnabled(Object.fromEntries(m.modules.map((x: any) => [x.id, x.enabled])));
    }).catch((e) => setErr(String(e)));
  }, []);

  const active = sessionState && ["initializing", "running", "stopping"].includes(sessionState);

  async function onUpload(f: File) {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.upload(f);
      setFile(r);
      setFilePath(r.path);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function start() {
    setBusy(true);
    setErr(null);
    try {
      const body: any = {
        mode, source, profile: profile === "auto" ? null : profile, save_media: saveMedia, composite_enabled: composite,
        modules: Object.entries(enabled).filter(([, v]) => v).map(([k]) => k),
        reference_text: mode === "reading" && refText.trim() ? refText : null,
        question: mode === "qa" && question.trim() ? question : null,
      };
      if (source === "file") body.file = { path: filePath, realtime, speed: 1 };
      else
        body.live = {
          monitor, fps, audio_device: audioDevice || null,
          region: region.full ? null : { left: region.left, top: region.top, width: region.width, height: region.height },
        };
      const s = await api.createSession(body);
      await api.start(s.id);
      onStarted(s.id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!sessionId) return;
    setBusy(true);
    try {
      await api.stop(sessionId);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const mon = devices.monitors[monitor];
  return (
    <div className="grid2">
      <section className="card">
        <h3>Сессия</h3>
        <label>Режим
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} disabled={!!active}>
            <option value="reading">Reading — чтение произведения</option>
            <option value="qa">Q&A — ответ на вопрос</option>
          </select>
        </label>
        <label>Источник
          <select value={source} onChange={(e) => setSource(e.target.value as any)} disabled={!!active}>
            <option value="file">File Replay (MP4/MKV/WAV)</option>
            <option value="live">Live: область экрана + системный звук</option>
          </select>
        </label>
        {source === "file" ? (
          <>
            <label>Медиафайл
              <input type="file" accept=".mp4,.mkv,.wav,.mov,.webm,.mp3,.flac" disabled={!!active || busy} onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
            </label>
            <label>или локальный путь
              <input value={filePath} onChange={(e) => setFilePath(e.target.value)} placeholder="D:\\video\\reading.mp4" disabled={!!active} />
            </label>
            {file?.probe && <div className="hint">длительность {file.probe.duration_s?.toFixed(1)} с; видео {file.probe.video ? `${file.probe.video.width}×${file.probe.video.height}` : "нет"}; аудио {file.probe.audio ? `${file.probe.audio.rate} Гц` : "нет"}</div>}
            <label className="row"><input type="checkbox" checked={realtime} onChange={(e) => setRealtime(e.target.checked)} /> воспроизведение с реальной скоростью</label>
          </>
        ) : (
          <>
            <label>Монитор
              <select value={monitor} onChange={(e) => setMonitor(+e.target.value)}>
                {devices.monitors.map((m, i) => <option key={i} value={i}>{`#${i} ${m.width}×${m.height}${m.primary ? " (основной)" : ""}`}</option>)}
              </select>
            </label>
            <label className="row"><input type="checkbox" checked={region.full} onChange={(e) => setRegion({ ...region, full: e.target.checked })} /> весь экран</label>
            {!region.full && (
              <div className="row4">
                {(["left", "top", "width", "height"] as const).map((k) => (
                  <label key={k}>{k}<input type="number" value={region[k]} min={0} onChange={(e) => setRegion({ ...region, [k]: +e.target.value })} /></label>
                ))}
              </div>
            )}
            {mon && !region.full && (region.left + region.width > mon.width || region.top + region.height > mon.height) && <div className="warn">Область выходит за пределы монитора {mon.width}×{mon.height}</div>}
            <label>Системное аудиоустройство (loopback)
              <select value={audioDevice} onChange={(e) => setAudioDevice(e.target.value)}>
                <option value="">по умолчанию</option>
                {devices.audio_outputs.map((d) => <option key={d.id} value={d.name}>{d.name}{d.default ? " (по умолч.)" : ""}</option>)}
              </select>
            </label>
            <label>FPS захвата<input type="number" min={1} max={60} value={fps} onChange={(e) => setFps(+e.target.value)} /></label>
            <label className="row"><input type="checkbox" checked={saveMedia} onChange={(e) => setSaveMedia(e.target.checked)} /> сохранять исходную медиазапись</label>
          </>
        )}
        {mode === "reading" ? (
          <>
            <label>Эталонный текст (TXT)
              <input type="file" accept=".txt" onChange={async (e) => { const f = e.target.files?.[0]; if (f) setRefText(await f.text()); }} />
            </label>
            <textarea rows={6} value={refText} onChange={(e) => setRefText(e.target.value)} placeholder="Текст произведения (необязательно; без эталона WER не считается)" />
          </>
        ) : (
          <label>Вопрос участнику<textarea rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} /></label>
        )}
      </section>
      <section className="card">
        <h3>Модули</h3>
        {modules.map((m) => (
          <label key={m.id} className="row">
            <input type="checkbox" checked={!!enabled[m.id]} disabled={!!active} onChange={(e) => setEnabled({ ...enabled, [m.id]: e.target.checked })} />
            {MODULE_TITLES[m.id] ?? m.title}
          </label>
        ))}
        <label>Профиль производительности
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            <option value="auto">auto (GPU при наличии CUDA)</option>
            <option value="gpu">gpu — RTX 3080</option>
            <option value="cpu">cpu</option>
          </select>
        </label>
        <label className="row"><input type="checkbox" checked={composite} onChange={(e) => setComposite(e.target.checked)} /> exploratory_composite_score (экспериментальный, невалидированный)</label>
        <div className="buttons">
          <button className="primary" disabled={busy || !!active || (source === "file" && !filePath)} onClick={start}>Start</button>
          <button disabled={busy || !active} onClick={stop}>Stop</button>
          <button disabled={busy || !!active} onClick={() => { setFile(null); setFilePath(""); setRefText(""); setQuestion(""); onReset(); }}>Reset</button>
        </div>
        {sessionId && <div className="hint">Сессия {sessionId}: {sessionState}</div>}
        {err && <div className="error">{err}</div>}
        <p className="hint">Захват экрана и звука начинается только после Start. Данные не покидают компьютер.</p>
      </section>
    </div>
  );
}
