#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
. "$ROOT_DIR/scripts/_python_env.sh"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

ITERATIONS="${1:-2}"
MAX_MINUTES="${2:-20}"
SOLVER_ROUNDS="${3:-1}"
"$PYTHON" -m protein_agent_tiny.agent_runner \
  --iterations "$ITERATIONS" \
  --max-minutes "$MAX_MINUTES" \
  --solver-rounds "$SOLVER_ROUNDS"
"$PYTHON" -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
"$PYTHON" -m protein_agent_tiny.report --run-dir outputs/latest

echo "output.zip: $ROOT_DIR/outputs/latest/output.zip"
