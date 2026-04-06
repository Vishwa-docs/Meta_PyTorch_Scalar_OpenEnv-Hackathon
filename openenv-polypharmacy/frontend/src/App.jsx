import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://localhost:7860";
const WS_URL = "ws://localhost:7860/ws";
const TASKS = ["easy_screening", "budgeted_screening", "complex_tradeoff"];

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

export default function App() {
  const [taskId, setTaskId] = useState("budgeted_screening");
  const [obs, setObs] = useState(null);
  const [log, setLog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [action, setAction] = useState({
    action_type: "query_ddi",
    drug_id_1: "",
    drug_id_2: "",
    target_drug_id: "",
    intervention_type: "stop",
    proposed_new_drug_id: "",
    rationale: "",
  });

  const medIds = useMemo(
    () => (obs?.current_medications || []).map((m) => m.drug_id),
    [obs]
  );
  const hasValidEpisode = Boolean(obs?.episode_id) && (obs?.current_medications?.length || 0) > 0;
  const isDone = Boolean(obs?.done);
  const finalScore =
    typeof obs?.metadata?.grader_score === "number" ? obs.metadata.grader_score : null;
  const noBudgetsLeft =
    hasValidEpisode &&
    (obs?.remaining_query_budget ?? 0) <= 0 &&
    (obs?.remaining_intervention_budget ?? 0) <= 0;
  const wsRef = useRef(null);
  const pendingRef = useRef([]);

  const wsEnsure = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return wsRef.current;
    if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
      await new Promise((r) => setTimeout(r, 80));
      return wsEnsure();
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        const pending = pendingRef.current.shift();
        if (pending) pending.resolve(msg);
      } catch (e) {
        const pending = pendingRef.current.shift();
        if (pending) pending.reject(e);
      }
    };
    ws.onerror = (err) => {
      const pending = pendingRef.current.shift();
      if (pending) pending.reject(err);
    };
    ws.onclose = () => {
      wsRef.current = null;
    };

    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("WebSocket connect timeout")), 2500);
      ws.onopen = () => {
        clearTimeout(t);
        resolve();
      };
    });
    return ws;
  };

  const wsSend = async (type, data) => {
    const ws = await wsEnsure();
    return await new Promise((resolve, reject) => {
      pendingRef.current.push({ resolve, reject });
      ws.send(JSON.stringify({ type, data }));
    });
  };

  useEffect(() => {
    return () => {
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
    };
  }, []);

  const appendLog = (text) => {
    setLog((prev) => [`${new Date().toLocaleTimeString()}  ${text}`, ...prev].slice(0, 20));
  };

  const normalizeObsFromWs = (packetData) => {
    const observation = packetData?.observation || {};
    const mergedMetadata = {
      ...(observation?.metadata || {}),
      ...(packetData?.info || {}),
    };
    return {
      ...observation,
      done: Boolean(packetData?.done ?? observation?.done ?? false),
      reward: packetData?.reward ?? observation?.reward ?? null,
      metadata: mergedMetadata,
    };
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      const msg = await wsSend("reset", { task_id: taskId });
      const data = msg?.data || {};
      const normalized = normalizeObsFromWs(data);
      setObs(normalized);
      const ids = (normalized?.current_medications || []).map((m) => m.drug_id);
      setAction((prev) => ({
        ...prev,
        drug_id_1: ids[0] || "",
        drug_id_2: ids[1] || "",
        target_drug_id: ids[0] || "",
      }));
      appendLog(`Reset task=${taskId}`);
    } catch (err) {
      appendLog(`Reset failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const buildActionPayload = () => {
    if (noBudgetsLeft) {
      return { action_type: "finish_review" };
    }
    if (action.action_type === "query_ddi") {
      return {
        action_type: "query_ddi",
        drug_id_1: action.drug_id_1,
        drug_id_2: action.drug_id_2,
      };
    }
    if (action.action_type === "propose_intervention") {
      return {
        action_type: "propose_intervention",
        target_drug_id: action.target_drug_id,
        intervention_type: action.intervention_type,
        proposed_new_drug_id: action.proposed_new_drug_id || undefined,
        rationale: action.rationale || undefined,
      };
    }
    return { action_type: "finish_review" };
  };

  const isActionValid = () => {
    if (!hasValidEpisode) return false;
    if (isDone) return false;
    if (noBudgetsLeft) return true;
    if (action.action_type === "query_ddi") {
      return Boolean(action.drug_id_1 && action.drug_id_2);
    }
    if (action.action_type === "propose_intervention") {
      return Boolean(action.target_drug_id && action.intervention_type);
    }
    return true;
  };

  const handleStep = async (overrideAction = null) => {
    if (!hasValidEpisode) {
      appendLog("Run Reset Episode before stepping.");
      return;
    }
    setLoading(true);
    try {
      const payload = overrideAction || buildActionPayload();
      const msg = await wsSend("step", payload);
      const data = msg?.data || {};
      const normalized = normalizeObsFromWs(data);
      setObs(normalized);
      appendLog(`Step: ${payload.action_type} -> reward=${data.reward ?? 0}`);
    } catch (err) {
      appendLog(`Step failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const askAi = async () => {
    if (!hasValidEpisode) {
      appendLog("Run Reset Episode before asking AI.");
      return;
    }
    setLoading(true);
    try {
      const data = await apiPost("/agent/suggest", { observation: obs });
      appendLog(`AI suggestion: ${data.action.action_type}`);
      await handleStep(data.action);
    } catch (err) {
      appendLog(`AI suggestion failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <div className="container">
      <header className="topbar glass">
        <div className="title-wrap">
          <h1>Polypharmacy Control Center</h1>
        </div>
        <div className={`status-chip ${hasValidEpisode ? "live" : "idle"}`}>
          {hasValidEpisode ? "Session Live" : "Waiting for reset"}
        </div>
        <div className="actions">
          <select value={taskId} onChange={(e) => setTaskId(e.target.value)}>
            {TASKS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <button onClick={handleReset} disabled={loading}>
            Reset Episode
          </button>
          <button className="secondary" onClick={askAi} disabled={!hasValidEpisode || isDone || loading}>
            Ask AI + Auto Step
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="panel glass panel-wide">
          <h2>Episode</h2>
          {hasValidEpisode ? (
            <div className="kpi-grid">
              <div><span>Episode</span><strong>{obs.episode_id}</strong></div>
              <div><span>Task</span><strong>{obs.task_id}</strong></div>
              <div><span>Age / Sex</span><strong>{obs.age} / {obs.sex}</strong></div>
              <div><span>Step</span><strong>{obs.step_index}</strong></div>
              <div><span>Query budget</span><strong>{obs.remaining_query_budget}</strong></div>
              <div><span>Intervention budget</span><strong>{obs.remaining_intervention_budget}</strong></div>
            </div>
          ) : (
            <p className="muted">Start with Reset Episode. Until then, step actions are blocked.</p>
          )}
          {noBudgetsLeft && (
            <p className="muted budget-note">Query and intervention budgets are exhausted. Finish review to get final score.</p>
          )}
          {isDone && (
            <p className="muted budget-note">
              Episode complete
              {finalScore !== null ? ` • final score: ${finalScore.toFixed(3)}` : ""}.
              Click Reset Episode to start a new case.
            </p>
          )}
        </section>

        <section className="panel glass">
          <h2>Action Console</h2>
          <div className="action-row">
            <label>Action type</label>
            <select
              value={action.action_type}
              onChange={(e) => setAction((a) => ({ ...a, action_type: e.target.value }))}
            >
              <option value="query_ddi">query_ddi</option>
              <option value="propose_intervention">propose_intervention</option>
              <option value="finish_review">finish_review</option>
            </select>
          </div>

          {action.action_type === "query_ddi" && (
            <div className="stack stack-two">
              <input
                placeholder="drug_id_1"
                value={action.drug_id_1}
                onChange={(e) => setAction((a) => ({ ...a, drug_id_1: e.target.value }))}
              />
              <input
                placeholder="drug_id_2"
                value={action.drug_id_2}
                onChange={(e) => setAction((a) => ({ ...a, drug_id_2: e.target.value }))}
              />
            </div>
          )}

          {action.action_type === "propose_intervention" && (
            <div className="stack">
              <select
                value={action.target_drug_id}
                onChange={(e) => setAction((a) => ({ ...a, target_drug_id: e.target.value }))}
              >
                <option value="">Select target drug</option>
                {medIds.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
              <select
                value={action.intervention_type}
                onChange={(e) => setAction((a) => ({ ...a, intervention_type: e.target.value }))}
              >
                <option value="stop">stop</option>
                <option value="dose_reduce">dose_reduce</option>
                <option value="substitute">substitute</option>
                <option value="add_monitoring">add_monitoring</option>
              </select>
              <input
                placeholder="proposed_new_drug_id (optional)"
                value={action.proposed_new_drug_id}
                onChange={(e) =>
                  setAction((a) => ({ ...a, proposed_new_drug_id: e.target.value }))
                }
              />
              <input
                placeholder="rationale (optional)"
                value={action.rationale}
                onChange={(e) => setAction((a) => ({ ...a, rationale: e.target.value }))}
              />
            </div>
          )}
          <button onClick={() => handleStep()} disabled={!isActionValid() || loading}>
            {noBudgetsLeft ? "Finish Review" : "Submit Step"}
          </button>
        </section>

        <section className="panel glass">
          <h2>Current Medications</h2>
          <div className="med-grid">
            {(obs?.current_medications || []).map((m) => (
              <div key={m.drug_id} className="med-card">
                <strong>{m.drug_id}</strong>
                <p>{m.generic_name}</p>
                <small>{m.dose_mg} mg • {m.atc_class}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="panel glass">
          <h2>Event Log</h2>
          <div className="logs">
            {log.map((line, idx) => (
              <div key={idx}>{line}</div>
            ))}
          </div>
        </section>
      </main>
      </div>
    </div>
  );
}
