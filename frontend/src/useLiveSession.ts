import { useEffect, useRef, useState } from "react";
import type { AnalysisResult, SessionInfo, SessionStatus } from "./types";
import { mstr, mval } from "./types";

const KEEP_MS = 180_000; // окно хранения live-рядов в браузере

export interface LiveState {
  info: SessionInfo | null;
  status: SessionStatus | null;
  preview: string | null;
  prosody: { t: number; f0: number | null; rms: number | null }[];
  face: { t: number; smile: number | null; brow: number | null; mouth: number | null; intensity: number | null; presence: number | null }[];
  voice: AnalysisResult[];
  pauses: [number, number][];
  speech: { t0: number; t1: number }[];
  finals: { t0: number; t1: number; text: string; segment_id: number }[];
  partial: { text: string; segment_id: number } | null;
  reference: { sentence: string; sentence_index: number; t0: number } | null;
  text: AnalysisResult[];
  transactional: AnalysisResult[];
  errors: { module_id: string; message: string; t_ms: number; fatal: boolean }[];
  tmax: number;
  connected: boolean;
}

const empty: LiveState = {
  info: null, status: null, preview: null, prosody: [], face: [], voice: [], pauses: [], speech: [],
  finals: [], partial: null, reference: null, text: [], transactional: [], errors: [], tmax: 0, connected: false,
};

function trim<T>(arr: T[], t: (x: T) => number, tmax: number): T[] {
  const lim = tmax - KEEP_MS;
  let i = 0;
  while (i < arr.length && t(arr[i]) < lim) i++;
  return i ? arr.slice(i) : arr;
}

/** Подписка на WS сессии. Обновление состояния батчится (≈4 раза/с), чтобы UI не мешал inference. */
export function useLiveSession(sessionId: string | null): LiveState {
  const [state, setState] = useState<LiveState>(empty);
  const buf = useRef<any[]>([]);

  useEffect(() => {
    if (!sessionId) return;
    setState(empty);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(`${proto}://${location.host}/ws/sessions/${sessionId}`);
      ws.onopen = () => {
        setState((s) => ({ ...s, connected: true }));
        // история, накопленная до подключения (перезагрузка страницы / повторное подключение)
        fetch(`/api/sessions/${sessionId}/results`).then((r) => (r.ok ? r.json() : null)).then((d) => {
          if (!d?.results) return;
          const hist = d.results.filter((r: any) => r.module_id !== "fusion").map((r: any) => ({ type: "result", data: r }));
          setState((s) => reduce({ ...empty, info: s.info, status: s.status, preview: s.preview, connected: s.connected }, hist));
        }).catch(() => {});
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!closed) retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (e) => buf.current.push(JSON.parse(e.data));
    };
    connect();
    const timer = window.setInterval(() => {
      if (!buf.current.length) return;
      const msgs = buf.current;
      buf.current = [];
      setState((s) => reduce(s, msgs));
    }, 250);
    return () => {
      closed = true;
      window.clearInterval(timer);
      window.clearTimeout(retry);
      ws?.close();
    };
  }, [sessionId]);
  return state;
}

function reduce(s0: LiveState, msgs: any[]): LiveState {
  const s: LiveState = { ...s0 };
  for (const m of msgs) {
    const d = m.data;
    switch (m.type) {
      case "session":
        s.info = d;
        break;
      case "status":
        s.status = d;
        s.tmax = Math.max(s.tmax, d.media_ms ?? 0);
        break;
      case "preview":
        s.preview = d.jpeg_b64;
        break;
      case "transcript.partial":
        s.partial = { text: d.text, segment_id: d.segment_id };
        break;
      case "transcript.final":
        s.finals = [...s.finals.filter((f) => f.segment_id !== d.segment_id), { t0: d.t_start_ms, t1: d.t_end_ms, text: d.text, segment_id: d.segment_id }].slice(-200);
        if (s.partial?.segment_id === d.segment_id) s.partial = null;
        break;
      case "text.reference":
        s.reference = { sentence: d.sentence, sentence_index: d.sentence_index, t0: d.t_start_ms };
        break;
      case "speech.activity":
        if (d.kind === "end") s.speech = [...s.speech, { t0: d.t_start_ms, t1: d.t_end_ms }].slice(-300);
        break;
      case "module_error":
        s.errors = [...s.errors, d].slice(-100);
        break;
      case "result":
        addResult(s, d as AnalysisResult);
        break;
    }
  }
  s.prosody = trim(s.prosody, (x) => x.t, s.tmax);
  s.face = trim(s.face, (x) => x.t, s.tmax);
  s.voice = trim(s.voice, (x) => x.t_end_ms, s.tmax);
  s.pauses = trim(s.pauses, (x) => x[1], s.tmax);
  s.speech = trim(s.speech, (x) => x.t1, s.tmax);
  return s;
}

function addResult(s: LiveState, r: AnalysisResult) {
  s.tmax = Math.max(s.tmax, r.t_end_ms);
  switch (r.module_id) {
    case "prosody":
      if (r.payload.scope === "instantaneous") {
        s.prosody = [...s.prosody, { t: r.t_end_ms, f0: mval(r, "f0_mean_hz"), rms: mval(r, "rms_dbfs") }];
      }
      if (Array.isArray(r.payload.pauses) && r.payload.pauses.length) s.pauses = [...s.pauses, ...r.payload.pauses];
      break;
    case "face":
      s.face = [...s.face, {
        t: r.t_end_ms, smile: mval(r, "smile"), brow: mval(r, "brow_raise"), mouth: mval(r, "mouth_open"),
        intensity: mval(r, "expression_intensity"), presence: mval(r, "face_presence"),
      }];
      break;
    case "voice_emotion":
      if (r.status === "ok") s.voice = [...s.voice, r];
      break;
    case "speech":
      if (r.finality === "final" && r.status === "ok" && r.payload.text && r.payload.segment_id != null) {
        s.finals = [...s.finals.filter((f) => f.segment_id !== r.payload.segment_id),
          { t0: r.t_start_ms, t1: r.t_end_ms, text: r.payload.text, segment_id: r.payload.segment_id }]
          .sort((a, b) => a.t0 - b.t0).slice(-200);
      }
      break;
    case "text":
      s.text = [...s.text, r].slice(-50);
      break;
    case "transactional":
      s.transactional = [...s.transactional, r].slice(-50);
      break;
  }
}

export { mstr };
