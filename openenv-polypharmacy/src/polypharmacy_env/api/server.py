"""FastAPI server exposing the PolypharmacyEnv via OpenEnv HTTP endpoints.

Uses openenv.core.env_server.http_server.create_app to create a
standards-compliant OpenEnv server with WebSocket support.
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from ..env_core import PolypharmacyEnv
from ..models import PolypharmacyAction, PolypharmacyObservation

# Create the OpenEnv-compliant app using the framework's create_app.
# Pass the class (factory) so the server can create per-session instances.
app = create_app(
    PolypharmacyEnv,
    PolypharmacyAction,
    PolypharmacyObservation,
    env_name="polypharmacy_env",
)
