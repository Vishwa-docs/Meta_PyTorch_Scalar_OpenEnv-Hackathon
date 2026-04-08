"""FastAPI app factory for PolypharmacyEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
from starlette.responses import FileResponse

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyObservation
from .routes.agent import router as agent_router
from .routes.bandit import router as bandit_router

load_dotenv()


class SPAStaticFiles(StaticFiles):
    """Serve SPA index for unknown frontend routes."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code != 404:
            return response
        index_path = Path(self.directory) / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not Found")


# ── Stateful singleton for HTTP-based inference ──────────────────────────────
# OpenEnv's built-in HTTP /reset and /step handlers are stateless (they create
# a fresh env per call). The WebSocket /ws endpoint handles stateful sessions
# for the frontend. For the inference.py script (and the evaluator), we need
# HTTP endpoints that maintain state across reset → step → step → ... calls.
# We override OpenEnv's default routes with stateful versions.

_http_env: Optional[PolypharmacyEnv] = None


def _get_or_create_env() -> PolypharmacyEnv:
    global _http_env
    if _http_env is None:
        _http_env = PolypharmacyEnv()
    return _http_env


def _serialize_obs(obs: PolypharmacyObservation) -> Dict[str, Any]:
    """Convert observation to JSON-serializable dict."""
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()


def create_polypharmacy_app():
    app = create_app(
        PolypharmacyEnv,
        PolypharmacyAction,
        PolypharmacyObservation,
        env_name="polypharmacy_env",
    )

    # ── Override stateless HTTP routes with stateful ones ─────────────────

    # Remove OpenEnv's default /reset and /step routes so ours take priority
    new_routes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if path in ("/reset", "/step", "/state"):
            continue
        new_routes.append(route)
    app.routes[:] = new_routes

    @app.post("/reset")
    async def stateful_reset(body: Dict[str, Any] = {}):
        env = _get_or_create_env()
        task_id = body.get("task_id", None)
        kwargs = {}
        if task_id:
            kwargs["task_id"] = task_id
        seed = body.get("seed", None)
        episode_id = body.get("episode_id", None)
        obs = env.reset(seed=seed, episode_id=episode_id, **kwargs)
        obs_data = _serialize_obs(obs)
        return {
            "observation": obs_data,
            "reward": 0.0,
            "done": False,
        }

    @app.post("/step")
    async def stateful_step(body: Dict[str, Any] = {}):
        env = _get_or_create_env()
        action_data = body.get("action", body)
        try:
            action = PolypharmacyAction(**action_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        obs = env.step(action)
        obs_data = _serialize_obs(obs)
        # Extract metadata for top-level info
        metadata = obs_data.get("metadata", {}) or {}
        return {
            "observation": obs_data,
            "reward": obs_data.get("shaped_reward", 0.0),
            "done": obs_data.get("done", False),
            "info": metadata,
        }

    @app.get("/state")
    async def stateful_state():
        env = _get_or_create_env()
        state = env.state
        return state.model_dump() if hasattr(state, "model_dump") else state.dict()

    # ── Middleware & extra routes ─────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(agent_router)
    app.include_router(bandit_router)

    # In Docker Space deployment, serve built frontend from same container.
    project_root = Path(__file__).resolve().parents[4]
    frontend_dist = project_root / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", SPAStaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_polypharmacy_app()
