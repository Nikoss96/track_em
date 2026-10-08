import { useEffect, useState } from "react";
import { api } from "./api";
import Live from "./pages/Live";
import Results from "./pages/Results";
import Settings from "./pages/Settings";
import Setup from "./pages/Setup";
import { useLiveSession } from "./useLiveSession";

type Tab = "setup" | "live" | "results" | "settings";
const TABS: [Tab, string][] = [["setup", "Сессия"], ["live", "Live Dashboard"], ["results", "Результаты"], ["settings", "Настройки"]];

function fromHash(): { tab: Tab; id: string | null } {
  const m = location.hash.match(/^#\/(setup|live|results|settings)(?:\/([\w-]+))?/);
  return m ? { tab: m[1] as Tab, id: m[2] ?? null } : { tab: "setup", id: null };
}

export default function App() {
  const init = fromHash();
  const [tab, setTabState] = useState<Tab>(init.tab);
  const [sessionId, setSessionId] = useState<string | null>(init.tab === "live" ? init.id : null);
  const [resultsId, setResultsId] = useState<string | null>(init.tab === "results" ? init.id : null);
  const setTab = (t: Tab) => setTabState(t);
  useEffect(() => {
    const id = tab === "live" ? sessionId : tab === "results" ? resultsId : null;
    history.replaceState(null, "", `#/${tab}${id ? "/" + id : ""}`);
  }, [tab, sessionId, resultsId]);
  const [health, setHealth] = useState<any>(null);
  const live = useLiveSession(sessionId);
  const state = live.status?.state ?? live.info?.state ?? null;

  useEffect(() => {
    const f = () => api.health().then(setHealth).catch(() => setHealth(null));
    f();
    const t = window.setInterval(f, 5000);
    return () => window.clearInterval(t);
  }, []);
  useEffect(() => {
    if (state === "completed" && sessionId) setResultsId(sessionId);
  }, [state, sessionId]);

  return (
    <div className="app">
      <header>
        <h1>Multimodal Behavior Analyzer</h1>
        <nav>{TABS.map(([k, t]) => <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{t}</button>)}</nav>
        <div className="health">
          {health ? `backend ok · CUDA ${health.cuda ? "да" : "нет"} · RAM ${health.ram_used_gb}/${health.ram_total_gb} ГБ · процесс ${health.process_rss_mb} МБ${(health.gpus ?? []).map((g: any) => ` · ${g.name} VRAM ${g.memory_used_mb}/${g.memory_total_mb} МБ`).join("")}` : <span className="bad">backend недоступен</span>}
        </div>
      </header>
      <div className="banner">Исследовательский инструмент. Метрики описывают наблюдаемые сигналы и выходы моделей; они не устанавливают внутреннее состояние, искренность или личностные черты человека.</div>
      <main>
        {tab === "setup" && <Setup sessionId={sessionId} sessionState={state} onStarted={(id) => { setSessionId(id); setTab("live"); }} onReset={() => setSessionId(null)} />}
        {tab === "live" && (sessionId ? <Live live={live} /> : <div className="hint">Сессия не запущена</div>)}
        {tab === "results" && <Results sessionId={resultsId} onSelect={setResultsId} />}
        {tab === "settings" && <Settings />}
      </main>
    </div>
  );
}
