#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run scripts/deploy_uv.sh first." >&2
  exit 1
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

ROUNDS="${1:-2}"
MAX_MINUTES="${2:-20}"
.venv/bin/python -m protein_agent_tiny.agent_runner --rounds "$ROUNDS" --max-minutes "$MAX_MINUTES"
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
.venv/bin/python -m protein_agent_tiny.report --run-dir outputs/latest

echo "output.zip: $ROOT_DIR/outputs/latest/output.zip"
