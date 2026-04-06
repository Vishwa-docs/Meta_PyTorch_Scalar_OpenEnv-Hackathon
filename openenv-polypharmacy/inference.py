#!/usr/bin/env python3
"""Baseline LLM inference script for the PolypharmacyEnv.

Uses Groq's OpenAI-compatible Chat Completions API to drive an LLM agent through the
PolypharmacyEnv HTTP API.  Emits structured stdout logs in the
[START], [STEP], [END] format required by the OpenEnv evaluation spec.

Environment variables:
  GROQ_API_KEY          – required
  GROQ_BASE_URL         – optional (default: https://api.groq.com/openai/v1)
  GROQ_MODEL_NAME       – model to use (default: llama-3.1-8b-instant)
  POLYPHARMACY_ENV_URL  – environment HTTP base URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.environ.get("GROQ_API_KEY", "")
API_BASE = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ENV_URL = os.environ.get("POLYPHARMACY_ENV_URL", "http://localhost:7860")

TASKS = ["easy_screening", "budgeted_screening", "complex_tradeoff"]
EPISODES_PER_TASK = 5

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ── Logging helpers ──────────────────────────────────────────────────────────

def _log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, default=str)}", flush=True)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── Environment HTTP helpers ─────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Observation → prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical pharmacist AI assistant reviewing an elderly patient's medication regimen.
You must reduce drug-interaction risk and address Beers-criteria violations while minimising
unnecessary medication changes.

Available actions (respond with STRICT JSON, no extra text):
1. Query a drug pair for interactions:
   {"action_type": "query_ddi", "drug_id_1": "...", "drug_id_2": "..."}

2. Propose an intervention:
   {"action_type": "propose_intervention", "target_drug_id": "...",
    "intervention_type": "stop|dose_reduce|substitute|add_monitoring",
    "proposed_new_drug_id": "...(optional)", "rationale": "..."}

3. Finish the review:
   {"action_type": "finish_review"}

Respond with EXACTLY ONE JSON object per turn. No markdown, no explanation outside JSON.
"""


def _summarise_obs(obs: Dict[str, Any]) -> str:
    meds = obs.get("current_medications", [])
    med_summary = "; ".join(
        f"{m['drug_id']}({m['generic_name']},{m['dose_mg']}mg)"
        for m in meds
    )
    queries = obs.get("interaction_queries", [])
    q_summary = "; ".join(
        f"{q['drug_id_1']}+{q['drug_id_2']}={q.get('severity','?')}"
        for q in queries
    )
    interventions = obs.get("interventions", [])
    iv_summary = "; ".join(
        f"{iv['action_type']}({iv['target_drug_id']})"
        for iv in interventions
    )
    return (
        f"Patient: age={obs.get('age')}, sex={obs.get('sex')}, "
        f"conditions={obs.get('conditions')}, "
        f"eGFR={obs.get('eGFR_category')}, liver={obs.get('liver_function_category')}\n"
        f"Medications: {med_summary}\n"
        f"Queries so far: {q_summary or 'none'}\n"
        f"Interventions so far: {iv_summary or 'none'}\n"
        f"Remaining query budget: {obs.get('remaining_query_budget')}\n"
        f"Remaining intervention budget: {obs.get('remaining_intervention_budget')}\n"
        f"Step: {obs.get('step_index')}"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def _ask_llm(obs_summary: str) -> Dict[str, Any]:
    """Call the LLM and parse a PolypharmacyAction JSON."""
    try:
        chat_resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_summary},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        text = (chat_resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        _err(f"LLM parse error: {e}")
        return {"action_type": "finish_review"}


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        _err("GROQ_API_KEY is required")
        sys.exit(1)

    run_id = str(uuid.uuid4())[:8]

    for task_id in TASKS:
        task_scores: List[float] = []
        task_rewards: List[float] = []

        _log("START", {
            "run_id": run_id,
            "task_id": task_id,
            "model": MODEL,
            "episodes": EPISODES_PER_TASK,
        })

        for ep_idx in range(EPISODES_PER_TASK):
            reset_resp = env_reset(task_id)
            obs = reset_resp["observation"]
            done = reset_resp.get("done", False)
            episode_id = obs.get("episode_id", f"ep_{ep_idx}")
            total_reward = 0.0
            step_idx = 0

            while not done:
                obs_summary = _summarise_obs(obs)
                action_payload = _ask_llm(obs_summary)

                step_resp = env_step(action_payload)
                obs = step_resp["observation"]
                reward = step_resp.get("reward", 0.0)
                done = step_resp.get("done", False)
                total_reward += reward

                _log("STEP", {
                    "run_id": run_id,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "step_index": step_idx,
                    "observation_summary": obs_summary[:200],
                    "action_payload": action_payload,
                    "reward": reward,
                    "done": done,
                })

                step_idx += 1

            grader_score = step_resp.get("info", {}).get("grader_score", 0.0)
            task_scores.append(grader_score)
            task_rewards.append(total_reward)

        _log("END", {
            "run_id": run_id,
            "task_id": task_id,
            "episodes": EPISODES_PER_TASK,
            "avg_grader_score": sum(task_scores) / max(len(task_scores), 1),
            "avg_total_reward": sum(task_rewards) / max(len(task_rewards), 1),
            "per_episode_scores": task_scores,
        })

    _err("Inference complete.")


if __name__ == "__main__":
    main()
