"""HTTP request/response schemas for the OpenEnv-compliant API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = Field(default=None, max_length=255)


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = Field(default=None, gt=0)
    request_id: Optional[str] = Field(default=None, max_length=255)


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "healthy"
