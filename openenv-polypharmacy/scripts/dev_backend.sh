#!/usr/bin/env bash
set -euo pipefail

uvicorn backend.main:app --reload --host 0.0.0.0 --port 7860
