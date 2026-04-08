"""API routes for neural bandit predictions and risk screening."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/bandit", tags=["bandit"])


# Lazy-loaded module-level bandit instance
_bandit_instance = None
_bandit_config: Dict[str, Any] = {}


def _get_bandit():
    global _bandit_instance, _bandit_config
    if _bandit_instance is None:
        from ...neural_bandits import OptimNeuralTS

        n_drugs = _bandit_config.get("n_drugs", 33)
        _bandit_instance = OptimNeuralTS(
            input_dim=n_drugs,
            hidden=64,
            reg_lambda=1.0,
            exploration_factor=1.0,
            lr=0.01,
            train_epochs=50,
            warmup_steps=50,
            total_steps=500,
            retrain_every=10,
            de_population=16,
            de_crossover=0.9,
            de_weight=1.0,
            de_steps=8,
        )
    return _bandit_instance


class DrugComboRequest(BaseModel):
    drug_ids: List[str] = Field(..., description="List of drug IDs in the combination")


class RiskPrediction(BaseModel):
    predicted_rr: float = Field(..., description="Predicted relative risk (association measure)")
    lower_bound: float = Field(..., description="Lower confidence bound (mean - 3*std)")
    is_potentially_harmful: bool = Field(..., description="True if lower_bound > 1.1 threshold")
    n_models_in_ensemble: int = Field(..., description="Number of models in the ensemble")


class BanditMetrics(BaseModel):
    total_steps: int = 0
    warmup_steps: int = 0
    n_ensemble_models: int = 0
    avg_reward: float = 0.0
    max_reward: float = 0.0
    phase: str = "not_started"


class ScreeningResult(BaseModel):
    drug_ids: List[str]
    predicted_rr: float
    lower_bound: float
    is_potentially_harmful: bool


class BulkScreenResponse(BaseModel):
    results: List[ScreeningResult]
    flagged_count: int
    total_screened: int


@router.post("/predict", response_model=RiskPrediction)
def predict_risk(payload: DrugComboRequest) -> RiskPrediction:
    """Predict risk for a drug combination using the neural bandit ensemble.

    Uses the trained ensemble of models from OptimNeuralTS to estimate
    the relative risk (RR) for a given drug combination. A pessimistic
    lower confidence bound is used to minimize false positives.
    """
    try:
        import torch
        from ...data_loader import load_drug_metadata

        bandit = _get_bandit()
        metadata = load_drug_metadata()
        all_drug_ids = sorted(metadata.keys())

        # Build multi-hot vector
        x = torch.zeros(len(all_drug_ids))
        for drug_id in payload.drug_ids:
            if drug_id in all_drug_ids:
                idx = all_drug_ids.index(drug_id)
                x[idx] = 1.0

        result = bandit.predict_risk(x)
        return RiskPrediction(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/metrics", response_model=BanditMetrics)
def get_bandit_metrics() -> BanditMetrics:
    """Return current neural bandit training metrics."""
    try:
        bandit = _get_bandit()
        metrics = bandit.get_metrics()
        return BanditMetrics(**metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/screen", response_model=BulkScreenResponse)
def screen_combinations(payload: Dict[str, Any]) -> BulkScreenResponse:
    """Screen multiple drug combinations for potential risk.

    Body: { "combinations": [["DRUG_A", "DRUG_B"], ...] }
    """
    try:
        import torch
        from ...data_loader import load_drug_metadata

        combos = payload.get("combinations", [])
        if not combos:
            raise HTTPException(status_code=400, detail="No combinations provided")

        bandit = _get_bandit()
        metadata = load_drug_metadata()
        all_drug_ids = sorted(metadata.keys())

        results = []
        for drug_ids in combos:
            x = torch.zeros(len(all_drug_ids))
            for drug_id in drug_ids:
                if drug_id in all_drug_ids:
                    idx = all_drug_ids.index(drug_id)
                    x[idx] = 1.0

            pred = bandit.predict_risk(x)
            results.append(ScreeningResult(
                drug_ids=drug_ids,
                predicted_rr=pred["predicted_rr"],
                lower_bound=pred["lower_bound"],
                is_potentially_harmful=pred["is_potentially_harmful"],
            ))

        flagged = sum(1 for r in results if r.is_potentially_harmful)
        return BulkScreenResponse(
            results=results,
            flagged_count=flagged,
            total_screened=len(results),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
