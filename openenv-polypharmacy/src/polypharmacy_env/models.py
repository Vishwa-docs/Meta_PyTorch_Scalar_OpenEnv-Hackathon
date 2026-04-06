"""Pydantic models for the PolypharmacyEnv environment.

Extends OpenEnv base types (Action, Observation, State) and defines
auxiliary records for medications, interactions, and interventions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Auxiliary models ─────────────────────────────────────────────────────────

class MedicationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_id: str
    generic_name: str
    atc_class: str
    dose_mg: float
    frequency: str = "qd"
    route: str = "po"
    is_high_risk_elderly: bool = False
    beers_flags: List[str] = Field(default_factory=list)


class InteractionQueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_id_1: str
    drug_id_2: str
    severity: Optional[str] = None
    recommendation: Optional[str] = None
    risk_score: Optional[float] = None
    step_index: int = 0


class InterventionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_drug_id: str
    action_type: Literal["stop", "dose_reduce", "substitute", "add_monitoring"]
    proposed_new_drug_id: Optional[str] = None
    rationale: str = ""
    step_index: int = 0


# ── OpenEnv wire models ─────────────────────────────────────────────────────

class PolypharmacyAction(BaseModel):
    """Action sent by the agent each step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: Literal["query_ddi", "propose_intervention", "finish_review"]
    drug_id_1: Optional[str] = None
    drug_id_2: Optional[str] = None
    target_drug_id: Optional[str] = None
    intervention_type: Optional[
        Literal["stop", "dose_reduce", "substitute", "add_monitoring", "none"]
    ] = None
    proposed_new_drug_id: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolypharmacyObservation(BaseModel):
    """Observation returned to the agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str = ""
    task_id: str = "budgeted_screening"
    age: int = 65
    sex: str = "M"
    conditions: List[str] = Field(default_factory=list)
    eGFR_category: str = "normal"
    liver_function_category: str = "normal"
    current_medications: List[MedicationEntry] = Field(default_factory=list)
    interaction_queries: List[InteractionQueryRecord] = Field(default_factory=list)
    interventions: List[InterventionRecord] = Field(default_factory=list)
    step_index: int = 0
    remaining_query_budget: int = 0
    remaining_intervention_budget: int = 0
    shaped_reward: float = 0.0
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolypharmacyState(BaseModel):
    """Compact state snapshot for the /state endpoint."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: Optional[str] = None
    task_id: str = ""
    step_count: int = Field(default=0, ge=0)
    max_steps: int = 0
    num_query_actions: int = 0
    num_interventions: int = 0
