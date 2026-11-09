const BASE = (import.meta as any).env?.VITE_API_URL ?? "http://127.0.0.1:8000";
export async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}
