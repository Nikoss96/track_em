export type ResultStatus = "ok" | "partial" | "insufficient_data" | "unavailable" | "not_applicable" | "error";

export interface Metric {
  name: string;
  value: number | string | boolean | unknown[] | Record<string, unknown> | null;
  unit?: string | null;
  normalized_score?: number | null;
  confidence?: number | null;
  method: string;
  status: ResultStatus;
  kind: string;
  group?: string | null;
}

export interface AnalysisResult {
  session_id: string;
  module_id: string;
  model_id: string;
  model_version: string;
  t_start_ms: number;
  t_end_ms: number;
  metrics: Metric[];
  status: ResultStatus;
  quality: { input_ok: boolean; notes: string[] };
  latency_ms: number;
  error?: string | null;
  finality: "final" | "provisional";
  payload: Record<string, any>;
}

export interface ModuleHealth {
  module_id: string;
  state: "disabled" | "loading" | "running" | "idle" | "unavailable" | "error";
  model_id?: string | null;
  model_version?: string | null;
  detail?: string | null;
  processed: number;
  dropped: number;
  errors: number;
  queue_size: number;
  last_latency_ms?: number | null;
  avg_latency_ms?: number | null;
}

export interface CaptureStatus {
  t_ms: number;
  video: { frames: number; fps: number; size: [number, number] | null; blank_frames: number; corrupted: number; no_signal: boolean; capture_lag_ms: number };
  audio: { chunks: number; source_rate: number | null; source_channels: number | null; seconds: number; gaps: number; gap_ms: number; silence: boolean; silent_ms: number; no_signal: boolean; levels: [number, number][] };
  eof: boolean;
  errors: { where: string; message: string; fatal: boolean; t_ms: number }[];
}

export interface SessionStatus {
  session_id: string;
  state: string;
  t_ms: number;
  media_ms: number;
  capture: CaptureStatus | null;
  vad: { in_speech: boolean; segments: number; speech_ms: number; error: string | null; queue: number } | null;
  modules: ModuleHealth[];
  gate: { slots: number; active: number; waiting: number } | null;
  results: number;
}

export interface SessionInfo {
  id: string;
  state: string;
  mode: string;
  source: string;
  created_at: string;
  started_at?: string | null;
  stopped_at?: string | null;
  error?: string | null;
  config: Record<string, any>;
}

export function mval(r: AnalysisResult, name: string): number | null {
  const m = r.metrics.find((x) => x.name === name);
  return m && typeof m.value === "number" ? m.value : null;
}

export function mstr(r: AnalysisResult, name: string): string | null {
  const m = r.metrics.find((x) => x.name === name);
  return m && typeof m.value === "string" ? m.value : null;
}

export const MODULE_TITLES: Record<string, string> = {
  face: "Мимика",
  prosody: "Интонация",
  voice_emotion: "Голосовая экспрессия",
  speech: "Речь (ASR)",
  text: "Семантика текста",
  transactional: "Транзактный анализ",
  fusion: "Fusion",
};

export const STATE_RU: Record<string, string> = {
  running: "работает",
  loading: "загрузка",
  disabled: "отключён",
  unavailable: "недоступен",
  error: "ошибка",
  idle: "завершён",
};
