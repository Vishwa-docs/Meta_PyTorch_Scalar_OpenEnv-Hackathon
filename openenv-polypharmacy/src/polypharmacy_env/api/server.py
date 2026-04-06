"""FastAPI server exposing the PolypharmacyEnv via OpenEnv HTTP endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyState
from .schemas import (
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)

app = FastAPI(
    title="PolypharmacyEnv",
    description="OpenEnv environment for elderly polypharmacy medication-review safety.",
    version="0.1.0",
)

# Module-level environment instance (single-session for simplicity)
_env = PolypharmacyEnv()


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest | None = None) -> ResetResponse:
    """Reset the environment and start a new episode."""
    task_id = req.task_id if req else None
    seed = req.seed if req else None
    episode_id = req.episode_id if req else None

    obs = _env.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    return ResetResponse(
        observation=obs.model_dump(),
        reward=0.0,
        done=False,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Execute one step in the environment."""
    try:
        action = PolypharmacyAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    result = _env.step(action)
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"],
    )


@app.get("/state", response_model=PolypharmacyState)
def state() -> PolypharmacyState:
    """Return the current environment state snapshot."""
    return _env.state


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")
