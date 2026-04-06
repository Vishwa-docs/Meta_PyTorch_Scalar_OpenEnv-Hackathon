#!/usr/bin/env python3
"""LLM-based inference agent for PolypharmacyEnv.

Connects to a running OpenEnv server via WebSocket (using PolypharmacyClient)
and runs an LLM agent that reviews a patient's medication regimen.

Usage:
    # Start server first:
    #   uvicorn polypharmacy_env.api.server:app --port 7860

    # Then run inference:
    python scripts/inference.py --task easy_screening --seed 42
    python scripts/inference.py --task budgeted_screening --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from polypharmacy_env.client import PolypharmacyClient
from polypharmacy_env.models import PolypharmacyAction, PolypharmacyObservation

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]


def format_observation_for_llm(obs: PolypharmacyObservation) -> str:
    """Convert an observation to a human-readable prompt for the LLM."""
    lines = [
        f"Patient: {obs.age}yo {obs.sex}",
        f"Conditions: {', '.join(obs.conditions)}",
        f"eGFR: {obs.eGFR_category}, Liver: {obs.liver_function_category}",
        f"Step: {obs.step_index}",
        f"Query budget remaining: {obs.remaining_query_budget}",
        f"Intervention budget remaining: {obs.remaining_intervention_budget}",
        "",
        "Current Medications:",
    ]
    for med in obs.current_medications:
        flags = f" [BEERS: {', '.join(med.beers_flags)}]" if med.beers_flags else ""
        high_risk = " [HIGH RISK ELDERLY]" if med.is_high_risk_elderly else ""
        lines.append(
            f"  - {med.drug_id} ({med.generic_name}) {med.atc_class} "
            f"{med.dose_mg}mg{high_risk}{flags}"
        )

    if obs.interaction_queries:
        lines.append("")
        lines.append("DDI Queries So Far:")
        for q in obs.interaction_queries:
            lines.append(
                f"  - {q.drug_id_1} + {q.drug_id_2}: "
                f"severity={q.severity}, rec={q.recommendation}"
            )

    if obs.interventions:
        lines.append("")
        lines.append("Interventions So Far:")
        for iv in obs.interventions:
            lines.append(f"  - {iv.action_type} {iv.target_drug_id}: {iv.rationale}")

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a clinical pharmacist assistant reviewing an elderly patient's medication regimen.

Your goal: identify dangerous drug-drug interactions and Beers Criteria violations,
then propose safe interventions (stop, dose_reduce, substitute, add_monitoring) to
reduce risk while preserving therapeutic coverage.

Available actions (respond with JSON):
1. {"action_type": "query_ddi", "drug_id_1": "...", "drug_id_2": "..."}
   - Check for a drug-drug interaction between two medications.
2. {"action_type": "propose_intervention", "target_drug_id": "...", \
"intervention_type": "stop|dose_reduce|substitute|add_monitoring", "rationale": "..."}
   - Propose a change to the regimen.
3. {"action_type": "finish_review"}
   - End the review and submit your final regimen.

Strategy tips:
- Query high-risk drug pairs first (especially those flagged as high-risk elderly or Beers).
- Prioritise resolving severe DDIs over moderate ones.
- Prefer substitution over stopping when possible.
- Always provide a clinical rationale for interventions.
- Finish the review when you've addressed all major issues or exhausted your budget.

Respond with ONLY a valid JSON action object, no explanation outside the JSON.\
"""


def parse_llm_action(text: str) -> PolypharmacyAction:
    """Parse an LLM response into a PolypharmacyAction."""
    text = text.strip()
    # Extract JSON from markdown code blocks if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    data = json.loads(text)
    return PolypharmacyAction(**data)


async def run_llm_episode(
    base_url: str,
    task_id: str,
    seed: int,
    model: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Run a single episode with LLM agent via WebSocket."""
    if OpenAI is None:
        raise ImportError("openai package is required. Install with: pip install openai")

    llm = OpenAI()
    total_reward = 0.0
    steps = 0
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async with PolypharmacyClient(base_url=base_url) as client:
        result = await client.reset(task_id=task_id, seed=seed)
        obs = result.observation

        while not result.done:
            obs_text = format_observation_for_llm(obs)
            messages.append({"role": "user", "content": obs_text})

            # Call LLM
            action = None
            for attempt in range(max_retries):
                try:
                    response = llm.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=256,
                    )
                    llm_text = response.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": llm_text})
                    action = parse_llm_action(llm_text)
                    break
                except (json.JSONDecodeError, Exception) as e:
                    if attempt == max_retries - 1:
                        print(f"  LLM parse failed after {max_retries} attempts: {e}")
                        action = PolypharmacyAction(action_type="finish_review")
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"Invalid JSON. Please respond with only a valid JSON action. Error: {e}",
                        })

            assert action is not None
            result = await client.step(action)
            obs = result.observation
            total_reward += result.reward or 0.0
            steps += 1

            print(
                f"  step={steps} action={action.action_type} "
                f"reward={result.reward:.4f} done={result.done}"
            )

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": total_reward,
        "steps": steps,
    }


async def amain(args: argparse.Namespace) -> None:
    results = []
    for seed in range(args.seed, args.seed + args.episodes):
        print(f"\n=== Episode: task={args.task} seed={seed} ===")
        result = await run_llm_episode(
            base_url=args.url,
            task_id=args.task,
            seed=seed,
            model=args.model,
        )
        results.append(result)
        print(f"  => reward={result['total_reward']:.4f} steps={result['steps']}")

    if results:
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print(f"\nAverage reward over {len(results)} episodes: {avg_reward:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM agent on PolypharmacyEnv")
    parser.add_argument("--url", default="ws://localhost:7860", help="Server URL")
    parser.add_argument("--task", default="budgeted_screening", help="Task ID")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
