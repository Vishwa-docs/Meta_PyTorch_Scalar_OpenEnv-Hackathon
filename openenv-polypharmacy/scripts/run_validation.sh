#!/usr/bin/env bash
# Run validation: tests, server smoke test, and heuristic baseline
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Running unit tests ==="
PYTHONPATH=src python3 -m pytest src/polypharmacy_env/tests/ -v

echo ""
echo "=== Running heuristic baseline ==="
PYTHONPATH=src python3 -m polypharmacy_env.baselines.heuristic_agent

echo ""
echo "=== Validation complete ==="
