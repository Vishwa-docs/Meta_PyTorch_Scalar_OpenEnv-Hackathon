"""ASGI entrypoint for backend service in monorepo layout."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
SRC = BACKEND_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polypharmacy_env.api.app import app  # noqa: E402

__all__ = ["app"]
