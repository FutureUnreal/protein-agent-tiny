#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run scripts/bootstrap_server.sh first." >&2
  exit 1
fi

.venv/bin/python -m protein_agent_tiny.score "$@"
