"""Environment configuration constants and task parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # openenv-polypharmacy/
DATA_DIR = PROJECT_ROOT / "data"
LOOKUPS_DIR = DATA_DIR / "lookups"
PROCESSED_DIR = DATA_DIR / "processed"

DDI_RULES_CSV = LOOKUPS_DIR / "ddi_rules.csv"
BEERS_CRITERIA_CSV = LOOKUPS_DIR / "beers_criteria.csv"
DRUG_METADATA_CSV = LOOKUPS_DIR / "drug_metadata.csv"
PATIENTS_CSV = PROCESSED_DIR / "patients_polypharmacy.csv"

# ── Reward hyper-parameters ──────────────────────────────────────────────────
QUERY_COST: float = 0.01
INTERVENTION_COST: float = 0.02
INVALID_ACTION_PENALTY: float = 0.10
TIMEOUT_PENALTY: float = 0.20
SEVERE_DDI_DISCOVERY_BONUS: float = 0.03

# ── Task parameters ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    min_drugs: int
    max_drugs: int
    query_budget: int
    intervention_budget: int
    max_steps: int


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy_screening": TaskConfig(
        task_id="easy_screening",
        difficulty="easy",
        min_drugs=3,
        max_drugs=5,
        query_budget=4,
        intervention_budget=2,
        max_steps=10,
    ),
    "budgeted_screening": TaskConfig(
        task_id="budgeted_screening",
        difficulty="medium",
        min_drugs=6,
        max_drugs=10,
        query_budget=8,
        intervention_budget=3,
        max_steps=20,
    ),
    "complex_tradeoff": TaskConfig(
        task_id="complex_tradeoff",
        difficulty="hard",
        min_drugs=10,
        max_drugs=15,
        query_budget=12,
        intervention_budget=5,
        max_steps=30,
    ),
}

DEFAULT_TASK = "budgeted_screening"

# ── Critical drugs (must not be stopped without substitution) ────────────────
CRITICAL_DRUG_IDS: set[str] = {
    "DRUG_WARFARIN",
    "DRUG_APIXABAN",
    "DRUG_INSULIN_GLARGINE",
    "DRUG_METOPROLOL",
    "DRUG_DIGOXIN",
}
