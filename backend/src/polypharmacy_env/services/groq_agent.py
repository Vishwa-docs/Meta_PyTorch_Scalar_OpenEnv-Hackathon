"""Groq-powered action suggester for PolypharmacyEnv."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from ..models import PolypharmacyAction, PolypharmacyObservation

DEFAULT_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]
CRITICAL_DRUG_IDS = {"DRUG_WARFARIN", "DRUG_INSULIN_GLARGINE", "DRUG_DIGOXIN"}

SYSTEM_PROMPT = """You are a clinical medication safety assistant.
Return exactly one JSON object describing the next action.
Allowed output schema:
{
  "action_type": "query_ddi" | "propose_intervention" | "finish_review",
  "drug_id_1": "optional",
  "drug_id_2": "optional",
  "target_drug_id": "optional",
  "intervention_type": "stop|dose_reduce|substitute|add_monitoring|none",
  "proposed_new_drug_id": "optional",
  "rationale": "optional"
}
No markdown fences. No extra text.
Do NOT use finish_review early. First, gather evidence with query_ddi and/or
perform at least one meaningful intervention when needed.
"""


def _obs_to_prompt(obs: PolypharmacyObservation) -> str:
    meds = ", ".join(m.drug_id for m in obs.current_medications)
    conds = ", ".join(obs.conditions)
    return (
        f"Task: {obs.task_id}\n"
        f"Age: {obs.age}, sex: {obs.sex}\n"
        f"Conditions: {conds}\n"
        f"Medications: {meds}\n"
        f"Query budget: {obs.remaining_query_budget}\n"
        f"Intervention budget: {obs.remaining_intervention_budget}\n"
        f"Step index: {obs.step_index}\n"
        "Choose the single safest, most useful next action."
    )


def _parse_action(text: str) -> PolypharmacyAction:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    payload: dict[str, Any] = json.loads(raw)
    return PolypharmacyAction.model_validate(payload)


def _fallback_query_action(obs: PolypharmacyObservation) -> PolypharmacyAction:
    meds = [m.drug_id for m in obs.current_medications]
    if len(meds) >= 2 and obs.remaining_query_budget > 0:
        return PolypharmacyAction(
            action_type="query_ddi",
            drug_id_1=meds[0],
            drug_id_2=meds[1],
        )
    return PolypharmacyAction(action_type="finish_review")


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _pick_unseen_query_pair(obs: PolypharmacyObservation) -> tuple[str, str] | None:
    meds = [m.drug_id for m in obs.current_medications]
    if len(meds) < 2 or obs.remaining_query_budget <= 0:
        return None

    seen = {
        _norm_pair(q.drug_id_1, q.drug_id_2)
        for q in obs.interaction_queries
    }
    # Prioritize pairs containing high-risk drugs.
    high_risk = [m.drug_id for m in obs.current_medications if m.is_high_risk_elderly]
    ordered = high_risk + [m for m in meds if m not in set(high_risk)]

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            p = _norm_pair(ordered[i], ordered[j])
            if p not in seen:
                return p
    return None


def _pick_intervention_target(obs: PolypharmacyObservation) -> str | None:
    if obs.remaining_intervention_budget <= 0:
        return None
    med_set = {m.drug_id for m in obs.current_medications}

    # Use latest discovered severe/moderate query as intervention target.
    for q in reversed(obs.interaction_queries):
        if q.severity in ("severe", "moderate"):
            m1 = next((m for m in obs.current_medications if m.drug_id == q.drug_id_1), None)
            m2 = next((m for m in obs.current_medications if m.drug_id == q.drug_id_2), None)
            candidates = [m for m in (m1, m2) if m is not None]
            if not candidates:
                continue
            # Prefer non-critical risky drugs first.
            candidates.sort(
                key=lambda m: (
                    m.drug_id in CRITICAL_DRUG_IDS,
                    0 if any("avoid" in f for f in m.beers_flags) else 1,
                    0 if m.is_high_risk_elderly else 1,
                )
            )
            return candidates[0].drug_id

    # Fallback: if no severe/moderate discovered, still intervene on obviously
    # risky medications (Beers/high-risk flags) when budgets permit.
    risky = sorted(
        obs.current_medications,
        key=lambda m: (
            0 if any("avoid" in f for f in m.beers_flags) else 1,
            0 if m.is_high_risk_elderly else 1,
            1 if m.drug_id in CRITICAL_DRUG_IDS else 0,
        ),
    )
    for med in risky:
        if any("avoid" in f for f in med.beers_flags) or med.is_high_risk_elderly:
            return med.drug_id
    return None


def _rule_based_action(obs: PolypharmacyObservation) -> PolypharmacyAction | None:
    # If we already discovered significant risk, intervene before more querying.
    target = _pick_intervention_target(obs)
    if target and (
        obs.step_index >= 1
        and (
            obs.remaining_query_budget <= 2
            or len(obs.interaction_queries) >= 4
            or any(q.severity in ("severe", "moderate") for q in obs.interaction_queries)
        )
    ):
        intervention = "stop"
        rationale = "Remove likely contributor to discovered interaction risk"
        if target in CRITICAL_DRUG_IDS:
            # Avoid blunt stop for critical meds.
            intervention = "dose_reduce"
            rationale = "Critical medication: prefer dose reduction over abrupt stop"
        return PolypharmacyAction(
            action_type="propose_intervention",
            target_drug_id=target,
            intervention_type=intervention,
            rationale=rationale,
        )

    pair = _pick_unseen_query_pair(obs)
    if pair:
        return PolypharmacyAction(
            action_type="query_ddi",
            drug_id_1=pair[0],
            drug_id_2=pair[1],
        )

    if obs.remaining_intervention_budget > 0:
        # Final fallback before finish: at least one safety action.
        target = _pick_intervention_target(obs)
        if target:
            return PolypharmacyAction(
                action_type="propose_intervention",
                target_drug_id=target,
                intervention_type="dose_reduce"
                if target in CRITICAL_DRUG_IDS
                else "stop",
                rationale="Fallback intervention when query options are exhausted",
            )

    if obs.step_index >= 3:
        return PolypharmacyAction(action_type="finish_review")
    return None


def _postprocess_action(
    obs: PolypharmacyObservation, action: PolypharmacyAction
) -> PolypharmacyAction:
    # First apply deterministic guardrails to avoid repetitive loops.
    ruled = _rule_based_action(obs)
    if ruled is not None:
        return ruled

    # Guardrail: prevent useless immediate finish actions.
    if action.action_type == "finish_review":
        if obs.step_index < 2 and obs.remaining_query_budget > 0:
            return _fallback_query_action(obs)
        if len(obs.interaction_queries) == 0 and obs.remaining_query_budget > 0:
            return _fallback_query_action(obs)
    return action


def suggest_action_from_observation(
    observation: PolypharmacyObservation,
    model_name: str | None = None,
) -> PolypharmacyAction:
    """Use Groq chat completions to suggest a valid action."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
    model = (model_name or os.getenv("GROQ_MODEL_NAME", DEFAULT_MODEL)).strip()
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_prompt = _obs_to_prompt(observation)
    tried: list[tuple[str, str]] = []
    candidates: list[str] = [model] + [m for m in FALLBACK_MODELS if m != model]

    for candidate in candidates:
        try:
            resp = client.chat.completions.create(
                model=candidate,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=220,
            )
            generated = (resp.choices[0].message.content or "").strip()
            parsed = _parse_action(generated)
            return _postprocess_action(observation, parsed)
        except Exception as exc:
            tried.append((candidate, str(exc)))

    tried_txt = " | ".join(f"{m}: {err}" for m, err in tried)
    raise ValueError(
        "No Groq model worked. Try one of: "
        "llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it. "
        f"Errors: {tried_txt}"
    )
