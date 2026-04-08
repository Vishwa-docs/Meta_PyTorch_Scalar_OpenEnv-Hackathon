import { useEffect, useMemo, useRef, useState, useCallback } from "react";

function resolveApiBase() {
  const explicitBase = import.meta.env.VITE_API_BASE;
  if (explicitBase) return explicitBase.replace(/\/$/, "");

  const host = window.location.hostname;
  const isLocal =
    host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";

  if (isLocal && window.location.port === "5173") {
    return "http://localhost:7860";
  }
  return window.location.origin.replace(/\/$/, "");
}

const API_BASE = resolveApiBase();
const WS_URL = `${API_BASE.replace(/^http/, "ws")}/ws`;

const TASKS = [
  { id: "easy_screening", label: "Easy Screening" },
  { id: "budgeted_screening", label: "Budgeted Screening" },
  { id: "complex_tradeoff", label: "Complex Tradeoff" },
];

const TASK_LABEL_MAP = Object.fromEntries(TASKS.map((t) => [t.id, t.label]));

const ACTION_LABELS = {
  query_ddi: "Check Drug Interaction",
  propose_intervention: "Propose Change",
  finish_review: "Finish Review",
};

const INTERVENTION_LABELS = {
  stop: "Stop Medication",
  dose_reduce: "Reduce Dose",
  substitute: "Substitute with Safer Drug",
  add_monitoring: "Add Monitoring",
};

// ── Contextual guide steps: each targets a specific UI section ──────────────
const GUIDE_STEPS = [
  {
    target: "topbar",
    position: "below",
    title: "Welcome to PolypharmacyEnv",
    body: `This tool helps review elderly patients' medication regimens for safety.

You'll act as a pharmacist assistant: check pairs of drugs for harmful interactions, propose changes to reduce risk, and get scored on how well you protect the patient — all under limited budgets.

Behind the scenes, an AI agent (Neural Bandit) learns which drug combinations to investigate first, getting smarter with each review.`,
  },
  {
    target: "task-selector",
    position: "below",
    title: "Choose a Scenario",
    body: `Pick a difficulty level:

• Easy Screening — 3–5 drugs, 1 known dangerous interaction. Great for getting started.
• Budgeted Screening — 6–10 drugs, multiple problems to find, tighter budgets.
• Complex Tradeoff — 10–15 drugs including critical ones (blood thinners, insulin). Removing critical drugs without a replacement is penalized.

Click "Reset Episode" to load a new patient case.`,
  },
  {
    target: "episode-panel",
    position: "below",
    title: "Patient Overview",
    body: `After resetting, this panel shows the patient's details:

• Demographics (age, sex, medical conditions)
• Your remaining query and intervention budgets
• A risk bar comparing starting risk vs. current risk
• How many review steps you've taken

Each check and intervention uses up budget — use them wisely to get the best outcome.`,
  },
  {
    target: "action-console",
    position: "right",
    title: "Check Drug Interactions",
    body: `Select "Check Drug Interaction" and pick two drugs from the patient's list:

Example dangerous combinations:
• Warfarin + Naproxen → severe bleeding risk
• Diazepam + Tramadol → dangerous sedation
• Apixaban + Naproxen → severe bleeding risk

Each check costs a small amount of budget. Finding a serious interaction earns a bonus. A smart strategy checks high-risk pairs first.`,
  },
  {
    target: "action-console",
    position: "right",
    title: "Propose Changes",
    body: `After finding a dangerous interaction, switch to "Propose Change":

• Stop Medication — Remove the drug entirely
• Reduce Dose — Lower the dose to reduce risk
• Substitute Drug — Automatically finds a safer alternative in the same drug class
• Add Monitoring — Flag for closer clinical monitoring

Example: After finding warfarin + naproxen interaction, select Naproxen → "Substitute". The system finds a safer pain reliever.`,
  },
  {
    target: "medications-panel",
    position: "left",
    title: "Current Medications",
    body: `This grid shows the patient's active medications. Each card shows:

• Drug name and dose
• Drug class (e.g., pain reliever, blood thinner)
• "High Risk" badge for drugs that need extra caution in elderly patients
• Safety flags (avoid, caution, adjust dose)

Cards marked "avoid" or "High Risk" are prime candidates for a closer look. The list updates live as you make changes.`,
  },
  {
    target: "event-log",
    position: "above",
    title: "Activity Log & Score",
    body: `The log tracks every action you take and its impact. When you click "Finish Review", you get a final score (0–100%):

• Easy: Based on risk reduction + targeting the right dangerous drugs
• Medium: Risk reduction + precision of your interventions + how well you used your budget
• Hard: Risk reduction minus penalties for disrupting the patient's treatment plan

The "Ask AI" button lets an AI agent make decisions using the same tools you have.`,
  },
];

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

// ── Spotlight Guide Component ───────────────────────────────────────────────
function SpotlightGuide({ step, steps, onNext, onPrev, onClose }) {
  const [rect, setRect] = useState(null);
  const tooltipRef = useRef(null);

  const updateRect = useCallback(() => {
    const target = steps[step]?.target;
    if (!target) return;
    const el = document.querySelector(`[data-guide="${target}"]`);
    if (el) {
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      // scroll into view
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [step, steps]);

  useEffect(() => {
    updateRect();
    window.addEventListener("resize", updateRect);
    window.addEventListener("scroll", updateRect, true);
    return () => {
      window.removeEventListener("resize", updateRect);
      window.removeEventListener("scroll", updateRect, true);
    };
  }, [updateRect]);

  if (!rect) return null;

  const pad = 8;
  const current = steps[step];

  // Calculate tooltip position
  const getTooltipStyle = () => {
    const pos = current.position || "below";
    const base = {};
    if (pos === "below") {
      base.top = rect.top + rect.height + pad + 12;
      base.left = rect.left;
      base.maxWidth = Math.min(440, window.innerWidth - 40);
    } else if (pos === "above") {
      base.bottom = window.innerHeight - rect.top + pad + 12;
      base.left = rect.left;
      base.maxWidth = Math.min(440, window.innerWidth - 40);
    } else if (pos === "right") {
      base.top = rect.top;
      base.left = rect.left + rect.width + pad + 12;
      base.maxWidth = Math.min(380, window.innerWidth - rect.left - rect.width - 40);
    } else if (pos === "left") {
      base.top = rect.top;
      base.right = window.innerWidth - rect.left + pad + 12;
      base.maxWidth = Math.min(380, rect.left - 40);
    }
    return base;
  };

  return (
    <div className="spotlight-overlay">
      {/* Dark overlay with cutout */}
      <svg className="spotlight-svg" width="100%" height="100%">
        <defs>
          <mask id="spotlight-mask">
            <rect width="100%" height="100%" fill="white" />
            <rect
              x={rect.left - pad}
              y={rect.top - pad}
              width={rect.width + pad * 2}
              height={rect.height + pad * 2}
              rx="12"
              fill="black"
            />
          </mask>
        </defs>
        <rect
          width="100%"
          height="100%"
          fill="rgba(4, 6, 15, 0.75)"
          mask="url(#spotlight-mask)"
        />
      </svg>

      {/* Highlight border around target */}
      <div
        className="spotlight-ring"
        style={{
          top: rect.top - pad,
          left: rect.left - pad,
          width: rect.width + pad * 2,
          height: rect.height + pad * 2,
        }}
      />

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="spotlight-tooltip glass"
        style={getTooltipStyle()}
      >
        <div className="spotlight-tooltip-header">
          <h3>{current.title}</h3>
          <span className="guide-counter">
            {step + 1} / {steps.length}
          </span>
        </div>
        <div className="spotlight-tooltip-body">
          {current.body.split("\n").map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
        <div className="spotlight-tooltip-footer">
          <button
            className="guide-btn secondary"
            onClick={onPrev}
            disabled={step === 0}
          >
            Previous
          </button>
          <button className="guide-btn secondary" onClick={onClose}>
            Skip
          </button>
          {step < steps.length - 1 ? (
            <button className="guide-btn" onClick={onNext}>
              Next
            </button>
          ) : (
            <button className="guide-btn" onClick={onClose}>
              Done
            </button>
          )}
        </div>
        <div className="guide-dots">
          {steps.map((_, i) => (
            <span
              key={i}
              className={`dot ${i === step ? "active" : ""}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main App ────────────────────────────────────────────────────────────────

export default function App() {
  const [taskId, setTaskId] = useState("budgeted_screening");
  const [obs, setObs] = useState(null);
  const [log, setLog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [guideStep, setGuideStep] = useState(0);
  const [showGuide, setShowGuide] = useState(true);
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
  const hasValidEpisode =
    Boolean(obs?.episode_id) && (obs?.current_medications?.length || 0) > 0;
  const isDone = Boolean(obs?.done);
  const finalScore =
    typeof obs?.metadata?.grader_score === "number"
      ? obs.metadata.grader_score
      : null;
  const noBudgetsLeft =
    hasValidEpisode &&
    (obs?.remaining_query_budget ?? 0) <= 0 &&
    (obs?.remaining_intervention_budget ?? 0) <= 0;
  const wsRef = useRef(null);
  const pendingRef = useRef([]);

  const wsEnsure = async () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN)
      return wsRef.current;
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
      const t = setTimeout(
        () => reject(new Error("WebSocket connect timeout")),
        2500
      );
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
        /* ignore */
      }
    };
  }, []);

  const appendLog = (text) => {
    setLog((prev) =>
      [`${new Date().toLocaleTimeString()}  ${text}`, ...prev].slice(0, 30)
    );
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
      appendLog(`Reset — ${TASK_LABEL_MAP[taskId] || taskId}`);
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
      const label = ACTION_LABELS[payload.action_type] || payload.action_type;
      const rwd = data.reward ?? 0;
      appendLog(`${label} → reward: ${Number(rwd).toFixed(3)}`);
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
      const label =
        ACTION_LABELS[data.action.action_type] || data.action.action_type;
      appendLog(`AI suggests: ${label}`);
      await handleStep(data.action);
    } catch (err) {
      appendLog(`AI suggestion failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const formatDrugName = (drugId) => {
    if (!drugId) return "";
    return drugId
      .replace(/^DRUG_/, "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const currentRisk = obs?.metadata?.current_risk;
  const baselineRisk = obs?.metadata?.baseline_risk;

  return (
    <div className="shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      {/* Spotlight Guide */}
      {showGuide && (
        <SpotlightGuide
          step={guideStep}
          steps={GUIDE_STEPS}
          onNext={() => setGuideStep((s) => Math.min(s + 1, GUIDE_STEPS.length - 1))}
          onPrev={() => setGuideStep((s) => Math.max(0, s - 1))}
          onClose={() => setShowGuide(false)}
        />
      )}

      <div className="container">
        <header className="topbar glass" data-guide="topbar">
          <div className="title-wrap">
            <h1>PolypharmacyEnv</h1>
            <p>Elderly Medication Safety — Powered by Neural Bandits</p>
          </div>
          <div className="topbar-right">
            <div className={`status-chip ${hasValidEpisode ? "live" : "idle"}`}>
              {hasValidEpisode
                ? isDone
                  ? "Episode Complete"
                  : "Session Live"
                : "Ready"}
            </div>
            <button
              className="guide-trigger"
              onClick={() => {
                setGuideStep(0);
                setShowGuide(true);
              }}
              title="Open guided walkthrough"
            >
              ?
            </button>
          </div>
          <div className="actions" data-guide="task-selector">
            <select value={taskId} onChange={(e) => setTaskId(e.target.value)}>
              {TASKS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <button onClick={handleReset} disabled={loading}>
              Reset Episode
            </button>
            <button
              className="secondary"
              onClick={askAi}
              disabled={!hasValidEpisode || isDone || loading}
            >
              Ask AI + Auto Step
            </button>
          </div>
        </header>

        <main className="layout">
          {/* Episode Info */}
          <section className="panel glass panel-wide" data-guide="episode-panel">
            <h2>Episode Overview</h2>
            {hasValidEpisode ? (
              <>
                <div className="kpi-grid">
                  <div>
                    <span>Episode</span>
                    <strong>{obs.episode_id}</strong>
                  </div>
                  <div>
                    <span>Task</span>
                    <strong>{TASK_LABEL_MAP[obs.task_id] || obs.task_id}</strong>
                  </div>
                  <div>
                    <span>Patient</span>
                    <strong>
                      Age {obs.age}, {obs.sex === "M" ? "Male" : "Female"}
                    </strong>
                  </div>
                  <div>
                    <span>Step</span>
                    <strong>{obs.step_index}</strong>
                  </div>
                  <div>
                    <span>Query Budget</span>
                    <strong>{obs.remaining_query_budget} remaining</strong>
                  </div>
                  <div>
                    <span>Intervention Budget</span>
                    <strong>
                      {obs.remaining_intervention_budget} remaining
                    </strong>
                  </div>
                </div>

                {currentRisk !== undefined && baselineRisk !== undefined && (
                  <div className="risk-bar-wrap">
                    <div className="risk-labels">
                      <span>
                        Baseline Risk:{" "}
                        <strong>{Number(baselineRisk).toFixed(3)}</strong>
                      </span>
                      <span>
                        Current Risk:{" "}
                        <strong
                          className={
                            currentRisk < baselineRisk
                              ? "risk-down"
                              : "risk-same"
                          }
                        >
                          {Number(currentRisk).toFixed(3)}
                        </strong>
                      </span>
                    </div>
                    <div className="risk-bar">
                      <div
                        className="risk-fill"
                        style={{
                          width: `${Math.min(currentRisk * 100, 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {obs.conditions && obs.conditions.length > 0 && (
                  <div className="conditions-row">
                    <span className="conditions-label">Conditions:</span>
                    {obs.conditions.map((c) => (
                      <span key={c} className="condition-tag">
                        {c.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="muted">
                Select a task difficulty and click <strong>Reset Episode</strong>{" "}
                to begin a patient case.
              </p>
            )}
            {noBudgetsLeft && !isDone && (
              <div className="budget-note">
                All budgets exhausted. Click <strong>Finish Review</strong> to
                receive your final score.
              </div>
            )}
            {isDone && (
              <div className="budget-note done-note">
                Episode complete
                {finalScore !== null
                  ? ` — Final score: ${(finalScore * 100).toFixed(1)}%`
                  : ""}
                . Click <strong>Reset Episode</strong> to start a new case.
              </div>
            )}
          </section>

          {/* Action Console */}
          <section className="panel glass" data-guide="action-console">
            <h2>Action Console</h2>
            <div className="action-row">
              <label>Action Type</label>
              <select
                value={action.action_type}
                onChange={(e) =>
                  setAction((a) => ({ ...a, action_type: e.target.value }))
                }
              >
                {Object.entries(ACTION_LABELS).map(([val, label]) => (
                  <option key={val} value={val}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            {action.action_type === "query_ddi" && (
              <div className="stack stack-two">
                <div className="field-group">
                  <label>Drug 1</label>
                  <select
                    value={action.drug_id_1}
                    onChange={(e) =>
                      setAction((a) => ({ ...a, drug_id_1: e.target.value }))
                    }
                  >
                    <option value="">Select drug</option>
                    {medIds.map((id) => (
                      <option key={id} value={id}>
                        {formatDrugName(id)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-group">
                  <label>Drug 2</label>
                  <select
                    value={action.drug_id_2}
                    onChange={(e) =>
                      setAction((a) => ({ ...a, drug_id_2: e.target.value }))
                    }
                  >
                    <option value="">Select drug</option>
                    {medIds.map((id) => (
                      <option key={id} value={id}>
                        {formatDrugName(id)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {action.action_type === "propose_intervention" && (
              <div className="stack">
                <div className="field-group">
                  <label>Target Drug</label>
                  <select
                    value={action.target_drug_id}
                    onChange={(e) =>
                      setAction((a) => ({
                        ...a,
                        target_drug_id: e.target.value,
                      }))
                    }
                  >
                    <option value="">Select target drug</option>
                    {medIds.map((id) => (
                      <option key={id} value={id}>
                        {formatDrugName(id)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-group">
                  <label>Intervention Type</label>
                  <select
                    value={action.intervention_type}
                    onChange={(e) =>
                      setAction((a) => ({
                        ...a,
                        intervention_type: e.target.value,
                      }))
                    }
                  >
                    {Object.entries(INTERVENTION_LABELS).map(([val, label]) => (
                      <option key={val} value={val}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-group">
                  <label>New Drug ID (optional, for substitution)</label>
                  <input
                    placeholder="Leave blank for auto-selection"
                    value={action.proposed_new_drug_id}
                    onChange={(e) =>
                      setAction((a) => ({
                        ...a,
                        proposed_new_drug_id: e.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field-group">
                  <label>Rationale (optional)</label>
                  <input
                    placeholder="e.g., High bleeding risk with concurrent warfarin"
                    value={action.rationale}
                    onChange={(e) =>
                      setAction((a) => ({ ...a, rationale: e.target.value }))
                    }
                  />
                </div>
              </div>
            )}

            <button
              className="submit-btn"
              onClick={() => handleStep()}
              disabled={!isActionValid() || loading}
            >
              {noBudgetsLeft ? "Finish Review" : "Submit Step"}
            </button>
          </section>

          {/* Current Medications */}
          <section className="panel glass" data-guide="medications-panel">
            <h2>
              Current Medications
              {obs?.current_medications?.length
                ? ` (${obs.current_medications.length})`
                : ""}
            </h2>
            <div className="med-grid">
              {(obs?.current_medications || []).map((m) => (
                <div
                  key={m.drug_id}
                  className={`med-card ${m.is_high_risk_elderly ? "high-risk" : ""}`}
                >
                  <div className="med-card-header">
                    <strong>{formatDrugName(m.drug_id)}</strong>
                    {m.is_high_risk_elderly && (
                      <span className="risk-badge">High Risk</span>
                    )}
                  </div>
                  <p className="med-generic">{m.generic_name}</p>
                  <div className="med-details">
                    <span>{m.dose_mg} mg</span>
                    <span className="med-atc">{m.atc_class}</span>
                  </div>
                  {m.beers_flags && m.beers_flags.length > 0 && (
                    <div className="beers-flags">
                      {m.beers_flags.map((f, i) => (
                        <span key={i} className="beers-tag">
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
            {(!obs?.current_medications ||
              obs.current_medications.length === 0) && (
              <p className="muted">No medications loaded. Reset an episode to begin.</p>
            )}
          </section>

          {/* Interaction Queries & Interventions */}
          {hasValidEpisode && (
            <section className="panel glass panel-wide">
              <div className="history-grid">
                <div>
                  <h3>Drug Interaction Checks ({obs?.interaction_queries?.length || 0})</h3>
                  <div className="history-list">
                    {(obs?.interaction_queries || []).map((q, i) => (
                      <div
                        key={i}
                        className={`history-item severity-${q.severity}`}
                      >
                        <strong>
                          {formatDrugName(q.drug_id_1)} +{" "}
                          {formatDrugName(q.drug_id_2)}
                        </strong>
                        <span className={`severity-tag ${q.severity}`}>
                          {q.severity}
                        </span>
                        {q.recommendation && (
                          <p className="history-detail">
                            {q.recommendation.replace(/_/g, " ")}
                          </p>
                        )}
                      </div>
                    ))}
                    {(!obs?.interaction_queries || obs.interaction_queries.length === 0) && (
                      <p className="muted">No queries yet.</p>
                    )}
                  </div>
                </div>
                <div>
                  <h3>Proposed Changes ({obs?.interventions?.length || 0})</h3>
                  <div className="history-list">
                    {(obs?.interventions || []).map((iv, i) => (
                      <div key={i} className="history-item intervention-item">
                        <strong>{formatDrugName(iv.target_drug_id)}</strong>
                        <span className="intervention-tag">
                          {INTERVENTION_LABELS[iv.action_type] || iv.action_type}
                        </span>
                        {iv.proposed_new_drug_id && (
                          <p className="history-detail">
                            Replaced with: {formatDrugName(iv.proposed_new_drug_id)}
                          </p>
                        )}
                        {iv.rationale && (
                          <p className="history-detail">{iv.rationale}</p>
                        )}
                      </div>
                    ))}
                    {(!obs?.interventions || obs.interventions.length === 0) && (
                      <p className="muted">No interventions yet.</p>
                    )}
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* Event Log */}
          <section className="panel glass panel-wide" data-guide="event-log">
            <h2>Event Log</h2>
            <div className="logs">
              {log.length === 0 && (
                <div className="log-empty">
                  Events will appear here as you interact with the environment.
                </div>
              )}
              {log.map((line, idx) => (
                <div key={idx}>{line}</div>
              ))}
            </div>
          </section>
        </main>

        <footer className="app-footer">
          <p>
            PolypharmacyEnv — Built with{" "}
            <a
              href="https://github.com/meta-pytorch/OpenEnv"
              target="_blank"
              rel="noopener noreferrer"
            >
              PyTorch OpenEnv
            </a>{" "}
            | Based on{" "}
            <a
              href="https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5"
              target="_blank"
              rel="noopener noreferrer"
            >
              Neural Bandits for Polypharmacy
            </a>{" "}
            (Larouche et al.)
          </p>
        </footer>
      </div>
    </div>
  );
}
