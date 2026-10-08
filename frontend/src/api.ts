async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => req<any>("GET", "/api/health"),
  devices: () => req<{ monitors: any[]; audio_outputs: any[] }>("GET", "/api/devices"),
  modules: () => req<{ modules: any[]; models: any[]; profile: string }>("GET", "/api/modules"),
  config: () => req<any>("GET", "/api/config"),
  putConfig: (patch: any) => req<any>("PUT", "/api/config", patch),
  sessions: () => req<any[]>("GET", "/api/sessions"),
  createSession: (body: any) => req<any>("POST", "/api/sessions", body),
  start: (id: string) => req<any>("POST", `/api/sessions/${id}/start`),
  stop: (id: string) => req<any>("POST", `/api/sessions/${id}/stop`),
  session: (id: string) => req<any>("GET", `/api/sessions/${id}`),
  results: (id: string) => req<any>("GET", `/api/sessions/${id}/results`),
  async upload(file: File): Promise<{ path: string; probe?: any }> {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/files", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`${r.status}: ${(await r.json()).detail}`);
    return r.json();
  },
  exportUrl: (id: string, fmt: "json" | "csv" | "parquet") => `/api/sessions/${id}/export?format=${fmt}`,
};
