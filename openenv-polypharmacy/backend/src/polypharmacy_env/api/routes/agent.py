"""Agent suggestion API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...models import PolypharmacyAction, PolypharmacyObservation
from ...services.groq_agent import suggest_action_from_observation

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentSuggestRequest(BaseModel):
    observation: PolypharmacyObservation
    model_name: str | None = None


class AgentSuggestResponse(BaseModel):
    action: PolypharmacyAction
    source: str = Field(default="groq")


@router.post("/suggest", response_model=AgentSuggestResponse)
def suggest_agent_action(payload: AgentSuggestRequest) -> AgentSuggestResponse:
    """Return a model-suggested action for the current observation."""
    try:
        action = suggest_action_from_observation(
            payload.observation, model_name=payload.model_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model call failed: {exc}") from exc
    return AgentSuggestResponse(action=action)
