#!/usr/bin/env python3
"""Submission inference script for Polypharmacy OpenEnv environment.

Required environment variables:
  API_BASE_URL   OpenAI-compatible base URL
  MODEL_NAME     Model identifier
  HF_TOKEN       API key/token

Optional:
  POLYPHARMACY_ENV_URL  Environment API base (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
# Also accept GROQ_API_KEY or API_KEY as fallback for the token
_API_KEY = HF_TOKEN or os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
ENV_URL = os.getenv("POLYPHARMACY_ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "polypharmacy_env"
TASKS = ["easy_screening", "budgeted_screening", "complex_tradeoff"]
MAX_STEPS = 16
TEMPERATURE = 0.0
MAX_TOKENS = 220

VALID_ACTION_TYPES = {"query_ddi", "propose_intervention", "finish_review"}
VALID_INTERVENTIONS = {"stop", "dose_reduce", "substitute", "add_monitoring"}

SYSTEM_PROMPT = (
    "You are a clinical-pharmacist agent reviewing an elderly patient's medications. "
    "You MUST return ONLY a single JSON object (no markdown, no explanation). "
    "The action_type MUST be exactly one of: query_ddi, propose_intervention, finish_review. "
    "Schema for query_ddi: "
    '{"action_type":"query_ddi","drug_id_1":"DRUG_X","drug_id_2":"DRUG_Y"} '
    "Schema for propose_intervention: "
    '{"action_type":"propose_intervention","target_drug_id":"DRUG_X",'
    '"intervention_type":"stop|dose_reduce|substitute|add_monitoring",'
    '"rationale":"reason"} '
    "Schema for finish_review: "
    '{"action_type":"finish_review"} '
    "Strategy: First query_ddi for high-risk drug pairs (especially those with beers_flags). "
    "Then propose_intervention for dangerous findings. Finally finish_review."
)


def _b(v: bool) -> str:
    return str(bool(v)).lower()


def _fmt_reward(v: float) -> str:
    return f"{float(v):.2f}"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={_fmt_reward(reward)} "
        f"done={_b(done)} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(_fmt_reward(r) for r in rewards)
    print(
        f"[END] success={_b(success)} steps={steps} score={_clamp01(score):.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = text.replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _sanitize_action(data)
    except Exception:
        pass
    return {"action_type": "finish_review"}


def _sanitize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Build a clean action dict with only the fields relevant to the action type."""
    atype = raw.get("action_type", "")
    if atype not in VALID_ACTION_TYPES:
        return {"action_type": "finish_review"}

    if atype == "query_ddi":
        return {
            "action_type": "query_ddi",
            "drug_id_1": raw.get("drug_id_1") or None,
            "drug_id_2": raw.get("drug_id_2") or None,
        }
    if atype == "propose_intervention":
        it = raw.get("intervention_type", "")
        if it not in VALID_INTERVENTIONS:
            it = "add_monitoring"
        result: Dict[str, Any] = {
            "action_type": "propose_intervention",
            "target_drug_id": raw.get("target_drug_id") or None,
            "intervention_type": it,
        }
        new_drug = raw.get("proposed_new_drug_id") or None
        if new_drug:
            result["proposed_new_drug_id"] = new_drug
        rationale = raw.get("rationale") or None
        if rationale:
            result["rationale"] = rationale
        return result
    return {"action_type": "finish_review"}


def _llm_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    meds = obs.get("current_medications", [])
    summary = {
        "step_index": obs.get("step_index", 0),
        "remaining_query_budget": obs.get("remaining_query_budget", 0),
        "remaining_intervention_budget": obs.get("remaining_intervention_budget", 0),
        "conditions": obs.get("conditions", []),
        "current_medications": [
            {
                "drug_id": m.get("drug_id"),
                "generic_name": m.get("generic_name"),
                "dose_mg": m.get("dose_mg"),
                "beers_flags": m.get("beers_flags", []),
            }
            for m in meds
        ],
        "interaction_queries": obs.get("interaction_queries", []),
        "interventions": obs.get("interventions", []),
    }
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(summary, separators=(",", ":"))},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return _safe_json(content)


def _reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=45)
    r.raise_for_status()
    return r.json()


def _step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=45)
    if r.status_code == 422:
        # Invalid action — return a penalty and let the agent continue
        return {"observation": {}, "reward": -0.1, "done": False, "info": {"error": r.text[:200]}}
    r.raise_for_status()
    return r.json()


def run_task(client: OpenAI, task_id: str) -> None:
    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    log_start(task_id)
    try:
        reset_payload = _reset(task_id)
        obs = reset_payload.get("observation", {})
        done = bool(reset_payload.get("done", False))

        for i in range(1, MAX_STEPS + 1):
            if done:
                break
            action = _llm_action(client, obs)
            action_str = json.dumps(action, separators=(",", ":"))
            step_payload = _step(action)
            obs = step_payload.get("observation", {})
            reward = float(step_payload.get("reward") or 0.0)
            done = bool(step_payload.get("done", False))
            metadata = (obs or {}).get("metadata", {}) or {}
            last_error = metadata.get("error")
            rewards.append(reward)
            steps = i
            log_step(i, action_str, reward, done, str(last_error) if last_error else None)

            if done:
                raw_score = metadata.get("grader_score", None)
                if raw_score is not None:
                    score = _clamp01(float(raw_score))
                else:
                    score = _clamp01(sum(max(0.0, r) for r in rewards) / max(len(rewards), 1))
                success = score > 0.0
                break
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> int:
    if not _API_KEY:
        print("HF_TOKEN (or GROQ_API_KEY / API_KEY) is required", flush=True)
        return 1
    client = OpenAI(base_url=API_BASE_URL, api_key=_API_KEY)
    for task in TASKS:
        run_task(client, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
