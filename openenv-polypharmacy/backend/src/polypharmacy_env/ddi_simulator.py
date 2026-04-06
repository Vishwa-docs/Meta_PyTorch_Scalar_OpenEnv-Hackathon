"""Local DDI and guideline simulation using CSV lookup data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .data_loader import (
    BeersCriterion,
    DDIRule,
    DrugMeta,
    load_beers_criteria,
    load_ddi_rules,
    load_drug_metadata,
)


@dataclass(frozen=True)
class DDIResult:
    severity: str
    recommendation: str
    base_risk_score: float


_NO_INTERACTION = DDIResult(severity="none", recommendation="no_action", base_risk_score=0.0)


class DDISimulator:
    """Provides drug–drug interaction and Beers-criteria lookups."""

    def __init__(self) -> None:
        self._ddi_rules: Dict[Tuple[str, str], DDIRule] = load_ddi_rules()
        self._drug_meta: Dict[str, DrugMeta] = load_drug_metadata()
        self._beers: List[BeersCriterion] = load_beers_criteria()

    @staticmethod
    def _normalise_pair(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def lookup_ddi(self, drug_id_1: str, drug_id_2: str) -> DDIResult:
        key = self._normalise_pair(drug_id_1, drug_id_2)
        rule = self._ddi_rules.get(key)
        if rule is None:
            return _NO_INTERACTION
        return DDIResult(
            severity=rule.severity,
            recommendation=rule.recommendation,
            base_risk_score=rule.base_risk_score,
        )

    def get_beers_flags(
        self,
        drug_id: str,
        patient_conditions: List[str],
    ) -> List[str]:
        """Return list of Beers flags applicable to *drug_id* given patient conditions."""
        flags: List[str] = []
        for bc in self._beers:
            if bc.drug_id != drug_id:
                continue
            if bc.condition is None:
                flags.append(bc.criterion_type)
            elif bc.condition in patient_conditions:
                flags.append(f"{bc.criterion_type}_{bc.condition}")
        return flags

    def get_drug_meta(self, drug_id: str) -> Optional[DrugMeta]:
        return self._drug_meta.get(drug_id)

    def find_substitute(
        self,
        drug_id: str,
        current_drug_ids: List[str],
    ) -> Optional[str]:
        """Find a safer same-class substitute not already in the regimen."""
        meta = self._drug_meta.get(drug_id)
        if meta is None:
            return None
        candidates = [
            dm
            for dm in self._drug_meta.values()
            if (
                dm.atc_class == meta.atc_class
                and dm.drug_id != drug_id
                and dm.drug_id not in current_drug_ids
                and not dm.is_high_risk_elderly
            )
        ]
        if not candidates:
            return None
        # Pick the candidate with fewest severe DDIs with current regimen
        def _severe_count(cand: DrugMeta) -> int:
            count = 0
            for did in current_drug_ids:
                if did == drug_id:
                    continue
                r = self.lookup_ddi(cand.drug_id, did)
                if r.severity == "severe":
                    count += 1
            return count

        candidates.sort(key=lambda c: (_severe_count(c), c.drug_id))
        return candidates[0].drug_id

    @property
    def drug_metadata(self) -> Dict[str, DrugMeta]:
        return self._drug_meta

    @property
    def ddi_rules(self) -> Dict[Tuple[str, str], DDIRule]:
        return self._ddi_rules

    @property
    def beers_criteria(self) -> List[BeersCriterion]:
        return self._beers
