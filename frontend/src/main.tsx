import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { call } from "./services/api";

type Est = { estimator: string; estimand: string; estimate: number; se?: number;
  ci?: [number, number]; true_ate?: number; diagnostics: Record<string, unknown> };
const PAGES = ["Overview", "Datasets", "Causal Graphs", "Estimators", "Experiments",
  "Sensitivity", "Reports & Decisions"] as const;
const GRAPH = JSON.stringify({ nodes: [{ name: "X", role: "confounder" }, { name: "T", role: "treatment" },
  { name: "Y", role: "outcome" }], edges: [["X", "T"], ["X", "Y"], ["T", "Y"]] });
const SPEC = { synthetic: { n: 2000, seed: 1, confounding: 1.0 } };

function Json({ v }: { v: unknown }) { return <pre>{JSON.stringify(v, null, 2)}</pre>; }
function useRun<T>() {
  const [data, setData] = useState<T | null>(null); const [err, setErr] = useState<string>("");
  return { data, err, run: (p: Promise<T>) => { setErr(""); p.then(setData).catch(e => setErr(String(e))); } };
}

function Overview() {
  const h = useRun<{ status: string }>(); useEffect(() => h.run(call("/health")), []);
  return <div><h2>Overview</h2>{h.err ? <p>API unreachable: {h.err}</p> :
    <p>API status: {h.data?.status ?? "…"}. No experiments yet? Run one in the Estimators page —
    no placeholder numbers are shown.</p>}</div>;
}
function Datasets() {
  const d = useRun<unknown[]>(); useEffect(() => d.run(call("/datasets")), []);
  return <div><h2>Dataset registry</h2>{d.err && <p>{d.err}</p>}<Json v={d.data} /></div>;
}
function Graphs() {
  const r = useRun<unknown>();
  return <div><h2>Causal graph studio</h2><p>Example graph X→T, X→Y, T→Y</p>
    <button onClick={() => r.run(call("/causal/identify", { graph_json: GRAPH, treatment: "T", outcome: "Y" }))}>
      Identify adjustment sets</button>{r.err && <p>{r.err}</p>}<Json v={r.data} /></div>;
}
function Estimators() {
  const [est, setEst] = useState("aipw"); const r = useRun<Est>();
  return <div><h2>Estimator lab (synthetic SCM n=2000)</h2>
    <select value={est} onChange={e => setEst(e.target.value)}>
      {["naive", "regression_adjustment", "ipw", "aipw", "psm_att", "psm_ate"].map(x => <option key={x}>{x}</option>)}
    </select>
    <button onClick={() => r.run(call("/causal/estimate", { spec: SPEC, estimator: est }))}>Run</button>
    {r.err && <p>{r.err}</p>}{r.data && <><p>{r.data.estimand} = {r.data.estimate.toFixed(3)}
      {r.data.ci && ` [${r.data.ci[0].toFixed(3)}, ${r.data.ci[1].toFixed(3)}]`}
      {r.data.true_ate !== undefined && ` (true ${r.data.true_ate.toFixed(3)})`}</p><Json v={r.data.diagnostics} /></>}</div>;
}
function Experiments() { return <div><h2>Experiments</h2>
  <p>Benchmarks run offline via <code>scripts/run_experiments.py</code>; tracking artifacts are
  written to <code>artifacts/runs/</code> and MLflow (port 5000) when available.</p></div>; }
function Sensitivity() {
  const r = useRun<unknown>();
  return <div><h2>Sensitivity</h2><button onClick={() => r.run(call("/causal/sensitivity", { spec: SPEC }))}>
    Robustness value (OLS)</button>{r.err && <p>{r.err}</p>}<Json v={r.data} /></div>;
}
function Reports() {
  const [b, setB] = useState(0.2); const r = useRun<unknown>();
  return <div><h2>Policy evaluation</h2>
    <input type="number" step="0.1" min="0.1" max="1" value={b} onChange={e => setB(+e.target.value)} />
    <button onClick={() => r.run(call("/policies/evaluate", { spec: { synthetic: { n: 4000, seed: 3, randomized: true, nonlinear: true } }, budget: b }))}>
      Evaluate budget policy</button>{r.err && <p>{r.err}</p>}<Json v={r.data} /></div>;
}
const VIEW = [Overview, Datasets, Graphs, Estimators, Experiments, Sensitivity, Reports];
function App() {
  const [i, setI] = useState(0); const V = VIEW[i];
  return <div style={{ fontFamily: "sans-serif", margin: 24 }}><h1>CausalForge</h1>
    <nav>{PAGES.map((p, k) => <button key={p} onClick={() => setI(k)} disabled={k === i}>{p}</button>)}</nav><V /></div>;
}
createRoot(document.getElementById("root")!).render(<App />);
