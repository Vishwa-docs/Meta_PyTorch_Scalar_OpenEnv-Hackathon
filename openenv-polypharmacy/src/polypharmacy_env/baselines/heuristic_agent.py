"""Deterministic heuristic baseline agent for PolypharmacyEnv.

Strategy:
1. Query all unordered medication pairs for DDIs (within budget),
   prioritising high-risk elderly drugs first.
2. For each severe DDI found, attempt substitution or stop.
3. For each moderate DDI found, attempt substitution or stop.
4. For remaining budget, address Beers-flagged "avoid" drugs.
5. Call finish_review.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyObservation


def run_heuristic_episode(
    env: PolypharmacyEnv,
    task_id: str = "budgeted_screening",
    seed: int | None = None,
) -> Tuple[float, float, int]:
    """Run one episode with the heuristic agent.

    Returns (total_reward, grader_score, steps).
    """
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    grader_score = 0.0
    steps = 0

    # Phase 1: Query DDIs between medication pairs, prioritising high-risk drugs
    meds = obs.current_medications
    # Sort: high-risk elderly drugs first, then by Beers flag count
    meds_sorted = sorted(
        meds,
        key=lambda m: (not m.is_high_risk_elderly, -len(m.beers_flags), m.drug_id),
    )
    med_ids = [m.drug_id for m in meds_sorted]
    pairs: List[Tuple[str, str]] = list(combinations(med_ids, 2))
    severe_pairs: List[Tuple[str, str]] = []
    moderate_pairs: List[Tuple[str, str]] = []

    for a, b in pairs:
        if obs.remaining_query_budget <= 0:
            break
        action = PolypharmacyAction(
            action_type="query_ddi",
            drug_id_1=a,
            drug_id_2=b,
        )
        result = env.step(action)
        obs = PolypharmacyObservation(**result["observation"])
        total_reward += result["reward"]
        steps += 1

        if result["done"]:
            grader_score = result["info"].get("grader_score", 0.0)
            return total_reward, grader_score, steps

        # Track severity
        ddi_info = result["info"].get("ddi_result", {})
        sev = ddi_info.get("severity", "none")
        if sev == "severe":
            severe_pairs.append((a, b))
        elif sev == "moderate":
            moderate_pairs.append((a, b))

    # Phase 2: Intervene on severe DDI drugs first
    current_ids = [m.drug_id for m in obs.current_medications]
    intervened: set[str] = set()

    def _try_intervene(
        target: str,
        rationale: str,
    ) -> Tuple[bool, float, PolypharmacyObservation, int]:
        """Try substitute then stop. Returns (success, total_reward, obs, steps)."""
        nonlocal total_reward, steps
        # Try substitute first
        act = PolypharmacyAction(
            action_type="propose_intervention",
            target_drug_id=target,
            intervention_type="substitute",
            rationale=rationale,
        )
        res = env.step(act)
        obs_new = PolypharmacyObservation(**res["observation"])
        total_reward += res["reward"]
        steps += 1

        if res["done"]:
            return True, total_reward, obs_new, steps

        # If substitute failed, try stop
        if res["info"].get("warning"):
            if obs_new.remaining_intervention_budget <= 0:
                return False, total_reward, obs_new, steps
            act2 = PolypharmacyAction(
                action_type="propose_intervention",
                target_drug_id=target,
                intervention_type="stop",
                rationale=f"No substitute; {rationale}",
            )
            res2 = env.step(act2)
            obs_new = PolypharmacyObservation(**res2["observation"])
            total_reward += res2["reward"]
            steps += 1
            if res2["done"]:
                return True, total_reward, obs_new, steps

        return False, total_reward, obs_new, steps

    # Intervene on severe pairs
    for a, b in severe_pairs:
        if obs.remaining_intervention_budget <= 0:
            break
        # Pick the drug to intervene on (prefer the one not yet intervened)
        target = b if a in intervened else a
        if target in intervened:
            target = b
        if target in intervened:
            continue
        intervened.add(target)

        done, total_reward, obs, steps = _try_intervene(
            target, f"Severe DDI between {a} and {b}"
        )
        if done:
            grader_score = env._run_grader() if not done else 0.0
            # grader_score was already computed in step
            return total_reward, result["info"].get("grader_score", 0.0), steps

    # Phase 2b: Intervene on moderate DDI drugs
    for a, b in moderate_pairs:
        if obs.remaining_intervention_budget <= 0:
            break
        target = b if a in intervened else a
        if target in intervened:
            target = b
        if target in intervened:
            continue
        intervened.add(target)

        done, total_reward, obs, steps = _try_intervene(
            target, f"Moderate DDI between {a} and {b}"
        )
        if done:
            return total_reward, result["info"].get("grader_score", 0.0), steps

    # Phase 3: Address Beers-flagged "avoid" drugs
    for med in meds_sorted:
        if obs.remaining_intervention_budget <= 0:
            break
        if med.drug_id in intervened:
            continue
        if not med.beers_flags:
            continue
        if any("avoid" in f for f in med.beers_flags):
            intervened.add(med.drug_id)
            done, total_reward, obs, steps = _try_intervene(
                med.drug_id, f"Beers criteria: {', '.join(med.beers_flags)}"
            )
            if done:
                return total_reward, result["info"].get("grader_score", 0.0), steps

    # Phase 4: Finish
    action = PolypharmacyAction(action_type="finish_review")
    result = env.step(action)
    total_reward += result["reward"]
    steps += 1
    grader_score = result["info"].get("grader_score", 0.0)

    return total_reward, grader_score, steps


def run_heuristic_baseline(
    n_episodes: int = 5,
    task_ids: List[str] | None = None,
) -> None:
    """Run the heuristic agent across tasks and print results."""
    if task_ids is None:
        task_ids = ["easy_screening", "budgeted_screening", "complex_tradeoff"]

    env = PolypharmacyEnv()

    for tid in task_ids:
        scores: list[float] = []
        rewards: list[float] = []
        for i in range(n_episodes):
            total_r, score, steps = run_heuristic_episode(env, task_id=tid, seed=i)
            scores.append(score)
            rewards.append(total_r)
            print(f"  [{tid}] ep={i} steps={steps} reward={total_r:.4f} score={score:.4f}")

        avg_s = sum(scores) / len(scores) if scores else 0.0
        avg_r = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"  [{tid}] avg_score={avg_s:.4f}  avg_reward={avg_r:.4f}\n")


if __name__ == "__main__":
    run_heuristic_baseline()
