"""Deterministic graders for the three PolypharmacyEnv task difficulties."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from .data_loader import DDIRule
from .config import CRITICAL_DRUG_IDS
from .models import InterventionRecord


_EPS = 1e-8

# Scores must be strictly in (0.001, 0.999) — never outside this range
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clip(x: float) -> float:
    return max(_SCORE_MIN, min(x, _SCORE_MAX))


# ── Easy: easy_screening ─────────────────────────────────────────────────────

def grade_easy_screening(
    baseline_risk: float,
    final_risk: float,
    interventions: List[InterventionRecord],
    severe_ddi_drug_ids: List[Tuple[str, str]],
) -> float:
    """Score ∈ [0, 1] for the easy task.

    50 % risk reduction + 50 % targeted-intervention flag.
    """
    risk_reduction = max(0.0, baseline_risk - final_risk) / max(baseline_risk, _EPS)

    targeted = 0.0
    severe_drugs = set()
    for a, b in severe_ddi_drug_ids:
        severe_drugs.add(a)
        severe_drugs.add(b)
    for iv in interventions:
        if iv.target_drug_id in severe_drugs:
            targeted = 1.0
            break

    return _clip(0.5 * risk_reduction + 0.5 * targeted)


# ── Medium: budgeted_screening ───────────────────────────────────────────────

def grade_budgeted_screening(
    baseline_risk: float,
    final_risk: float,
    interventions: List[InterventionRecord],
    risk_deltas: List[float],
    num_queries: int,
    severe_moderate_discovered: int,
) -> float:
    """Score ∈ [0, 1] for the medium task.

    50 % risk reduction + 30 % intervention precision + 20 % query efficiency.
    """
    risk_reduction = max(0.0, baseline_risk - final_risk) / max(baseline_risk, _EPS)

    # Intervention precision: fraction of interventions that reduced risk
    if interventions:
        good = sum(1 for d in risk_deltas if d > 0)
        precision = good / len(interventions)
    else:
        precision = 0.0

    # Query efficiency
    if num_queries > 0:
        query_eff = min(severe_moderate_discovered / num_queries, 1.0)
    else:
        query_eff = 0.0

    return _clip(0.5 * risk_reduction + 0.3 * precision + 0.2 * query_eff)


# ── Hard: complex_tradeoff ───────────────────────────────────────────────────

def grade_complex_tradeoff(
    baseline_risk: float,
    final_risk: float,
    interventions: List[InterventionRecord],
    total_drug_changes: int,
    critical_drugs_stopped_without_sub: int,
) -> float:
    """Score ∈ [0, 1] for the hard task.

    Base = risk reduction; penalty for regimen disruption and critical-drug stops.
    """
    risk_reduction = max(0.0, baseline_risk - final_risk) / max(baseline_risk, _EPS)

    # Regimen disruption: penalise excessive changes
    disruption = 0.05 * total_drug_changes
    critical_penalty = 0.20 * critical_drugs_stopped_without_sub

    return _clip(risk_reduction - disruption - critical_penalty)
