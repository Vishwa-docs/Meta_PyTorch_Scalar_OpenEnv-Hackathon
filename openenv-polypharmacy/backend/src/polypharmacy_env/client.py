"""OpenEnv client for PolypharmacyEnv.

Provides a typed async/sync client for interacting with a PolypharmacyEnv
server via WebSocket, following the OpenEnv EnvClient pattern.

Example (async):
    >>> async with PolypharmacyClient(base_url="ws://localhost:8000") as env:
    ...     result = await env.reset(task_id="easy_screening")
    ...     result = await env.step(PolypharmacyAction(action_type="finish_review"))

Example (sync):
    >>> with PolypharmacyClient(base_url="ws://localhost:8000").sync() as env:
    ...     result = env.reset(task_id="easy_screening")
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PolypharmacyAction, PolypharmacyObservation, PolypharmacyState


class PolypharmacyClient(
    EnvClient[PolypharmacyAction, PolypharmacyObservation, PolypharmacyState]
):
    """Typed OpenEnv client for the PolypharmacyEnv environment."""

    def _step_payload(self, action: PolypharmacyAction) -> Dict[str, Any]:
        """Convert a PolypharmacyAction to the JSON payload for the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[PolypharmacyObservation]:
        """Parse a server response into a StepResult with typed observation."""
        obs_data = payload.get("observation", payload)
        obs = PolypharmacyObservation.model_validate(obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PolypharmacyState:
        """Parse a server state response into a typed PolypharmacyState."""
        return PolypharmacyState.model_validate(payload)
