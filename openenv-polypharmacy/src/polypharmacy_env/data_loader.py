"""Load and cache CSV lookup data for the PolypharmacyEnv."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    BEERS_CRITERIA_CSV,
    DDI_RULES_CSV,
    DRUG_METADATA_CSV,
    PATIENTS_CSV,
)


# ── Row-level data classes ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DrugMeta:
    drug_id: str
    generic_name: str
    atc_class: str
    is_high_risk_elderly: bool
    default_dose_mg: float
    min_dose_mg: float
    max_dose_mg: float


@dataclass(frozen=True)
class DDIRule:
    drug_id_1: str
    drug_id_2: str
    severity: str
    mechanism: str
    recommendation: str
    base_risk_score: float


@dataclass(frozen=True)
class BeersCriterion:
    drug_id: str
    criterion_type: str  # avoid | caution | dose_adjust | avoid_in_condition
    condition: Optional[str]
    rationale: str


@dataclass
class PatientEpisode:
    episode_id: str
    age: int
    sex: str
    conditions: List[str]
    eGFR_category: str
    liver_function_category: str
    medication_ids: List[str]
    baseline_risk_score: float
    difficulty: str


# ── Loaders (cached) ────────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_drug_metadata(path: Path = DRUG_METADATA_CSV) -> Dict[str, DrugMeta]:
    out: Dict[str, DrugMeta] = {}
    for row in _read_csv(path):
        dm = DrugMeta(
            drug_id=row["drug_id"],
            generic_name=row["generic_name"],
            atc_class=row["atc_class"],
            is_high_risk_elderly=row["is_high_risk_elderly"] == "1",
            default_dose_mg=float(row["default_dose_mg"]),
            min_dose_mg=float(row["min_dose_mg"]),
            max_dose_mg=float(row["max_dose_mg"]),
        )
        out[dm.drug_id] = dm
    return out


def _normalise_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


@lru_cache(maxsize=1)
def load_ddi_rules(path: Path = DDI_RULES_CSV) -> Dict[Tuple[str, str], DDIRule]:
    out: Dict[Tuple[str, str], DDIRule] = {}
    for row in _read_csv(path):
        key = _normalise_pair(row["drug_id_1"], row["drug_id_2"])
        out[key] = DDIRule(
            drug_id_1=key[0],
            drug_id_2=key[1],
            severity=row["severity"],
            mechanism=row["mechanism"],
            recommendation=row["recommendation"],
            base_risk_score=float(row["base_risk_score"]),
        )
    return out


@lru_cache(maxsize=1)
def load_beers_criteria(path: Path = BEERS_CRITERIA_CSV) -> List[BeersCriterion]:
    out: List[BeersCriterion] = []
    for row in _read_csv(path):
        cond = row["condition"].strip() or None
        out.append(BeersCriterion(
            drug_id=row["drug_id"],
            criterion_type=row["criterion_type"],
            condition=cond,
            rationale=row["rationale"],
        ))
    return out


def load_patients(
    path: Path = PATIENTS_CSV,
    difficulty: Optional[str] = None,
) -> List[PatientEpisode]:
    rows = _read_csv(path)
    eps: List[PatientEpisode] = []
    for row in rows:
        d = row.get("difficulty", "medium")
        if difficulty and d != difficulty:
            continue
        eps.append(PatientEpisode(
            episode_id=row["episode_id"],
            age=int(row["age"]),
            sex=row["sex"],
            conditions=[c.strip() for c in row["conditions"].split(";") if c.strip()],
            eGFR_category=row["eGFR_category"],
            liver_function_category=row["liver_function_category"],
            medication_ids=[m.strip() for m in row["medication_ids"].split(";") if m.strip()],
            baseline_risk_score=float(row["baseline_risk_score"]),
            difficulty=d,
        ))
    return eps
