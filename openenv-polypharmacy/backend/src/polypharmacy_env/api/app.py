"""FastAPI app factory for PolypharmacyEnv."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
from starlette.responses import FileResponse

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyObservation
from .routes.agent import router as agent_router

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


def create_polypharmacy_app():
    app = create_app(
        PolypharmacyEnv,
        PolypharmacyAction,
        PolypharmacyObservation,
        env_name="polypharmacy_env",
    )

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

    # In Docker Space deployment, serve built frontend from same container.
    project_root = Path(__file__).resolve().parents[4]
    frontend_dist = project_root / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", SPAStaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_polypharmacy_app()
