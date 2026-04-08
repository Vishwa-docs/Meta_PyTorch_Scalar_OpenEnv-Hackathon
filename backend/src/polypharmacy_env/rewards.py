"""Reward shaping and regimen-risk computation."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from .config import (
    INTERVENTION_COST,
    INVALID_ACTION_PENALTY,
    MODERATE_DDI_DISCOVERY_BONUS,
    QUERY_COST,
    SEVERE_DDI_DISCOVERY_BONUS,
    TIMEOUT_PENALTY,
)
from .data_loader import BeersCriterion, DDIRule, DrugMeta


def compute_regimen_risk(
    current_drug_ids: List[str],
    patient_conditions: List[str],
    ddi_rules: Dict[Tuple[str, str], DDIRule],
    beers_criteria: List[BeersCriterion],
    drug_metadata: Dict[str, DrugMeta],
) -> float:
    """Compute an aggregate risk score for the current medication regimen.

    Returns a float clipped to [0.0, 1.0].
    """
    if not current_drug_ids:
        return 0.0

    risk = 0.0
    drug_set = set(current_drug_ids)

    # 1. DDI pairwise risk
    for a, b in combinations(sorted(drug_set), 2):
        key = (a, b) if a < b else (b, a)
        rule = ddi_rules.get(key)
        if rule is not None:
            risk += rule.base_risk_score

    # 2. Beers violations (weights reflect clinical severity)
    beers_weight = {"avoid": 0.30, "caution": 0.12, "dose_adjust": 0.10, "avoid_in_condition": 0.25}
    for bc in beers_criteria:
        if bc.drug_id not in drug_set:
            continue
        if bc.condition is None:
            risk += beers_weight.get(bc.criterion_type, 0.05)
        elif bc.condition in patient_conditions:
            risk += beers_weight.get(bc.criterion_type, 0.05)

    # 3. High-risk elderly drugs
    for did in drug_set:
        dm = drug_metadata.get(did)
        if dm and dm.is_high_risk_elderly:
            risk += 0.05

    # Normalise by regimen size to keep score comparable across difficulties
    risk /= max(len(drug_set), 1)
    return min(max(risk, 0.0), 1.0)


def compute_shaped_reward(
    previous_risk: float,
    new_risk: float,
    action_type: str,
    *,
    is_invalid: bool = False,
    is_timeout: bool = False,
    discovered_severe: bool = False,
    discovered_moderate: bool = False,
) -> float:
    """Compute the step-level shaped reward."""
    reward = 0.0

    if is_invalid:
        return -INVALID_ACTION_PENALTY

    if is_timeout:
        return -TIMEOUT_PENALTY

    if action_type == "query_ddi":
        reward -= QUERY_COST
        if discovered_severe:
            reward += SEVERE_DDI_DISCOVERY_BONUS
        elif discovered_moderate:
            reward += MODERATE_DDI_DISCOVERY_BONUS

    elif action_type == "propose_intervention":
        reward += (previous_risk - new_risk)
        reward -= INTERVENTION_COST

    # finish_review terminal bonus is added by the caller after grading

    return max(0.0, reward)
