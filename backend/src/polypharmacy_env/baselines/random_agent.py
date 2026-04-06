"""Trivial random baseline agent for PolypharmacyEnv."""

from __future__ import annotations

import random
from typing import List, Tuple

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyObservation


def run_random_episode(
    env: PolypharmacyEnv,
    task_id: str = "budgeted_screening",
    seed: int | None = None,
) -> Tuple[float, float, int]:
    rng = random.Random(seed)
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    grader_score = 0.0
    steps = 0

    while not obs.done:
        med_ids = [m.drug_id for m in obs.current_medications]
        choice = rng.choice(["query_ddi", "propose_intervention", "finish_review"])

        if choice == "query_ddi" and len(med_ids) >= 2 and obs.remaining_query_budget > 0:
            pair = rng.sample(med_ids, 2)
            action = PolypharmacyAction(
                action_type="query_ddi",
                drug_id_1=pair[0],
                drug_id_2=pair[1],
            )
        elif choice == "propose_intervention" and med_ids and obs.remaining_intervention_budget > 0:
            target = rng.choice(med_ids)
            itype = rng.choice(["stop", "dose_reduce", "substitute", "add_monitoring"])
            action = PolypharmacyAction(
                action_type="propose_intervention",
                target_drug_id=target,
                intervention_type=itype,
                rationale="random",
            )
        else:
            action = PolypharmacyAction(action_type="finish_review")

        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps += 1
        if obs.done:
            grader_score = obs.metadata.get("grader_score", 0.0)
            break

    return total_reward, grader_score, steps
