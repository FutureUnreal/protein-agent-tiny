#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROUNDS="${1:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"

echo "== protein-agent-tiny bootstrap =="
echo "root: $ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv into the current user environment..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation did not put uv on PATH. Add ~/.local/bin to PATH and retry." >&2
  exit 1
fi

bash scripts/deploy_uv.sh

if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example"
else
  echo "kept existing .env"
fi

mkdir -p outputs workspaces runs memory

echo
echo "== environment status =="
".venv/bin/python" - <<'PY'
import os
from pathlib import Path

env = {}
path = Path(".env")
if path.exists():
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

for key in (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "PROTEIN_AGENT_MODEL",
    "OPENALEX_API_KEY",
    "PROTEIN_AGENT_MAX_INPUT_TOKENS",
    "PROTEIN_AGENT_MAX_OUTPUT_TOKENS",
):
    value = os.environ.get(key) or env.get(key, "")
    if "KEY" in key:
        status = "set" if value else "missing"
        print(f"{key}: {status}")
    else:
        print(f"{key}: {value or 'missing'}")
PY

if [ "$RUN_BASELINE" = "1" ]; then
  echo
  echo "== baseline smoke test =="
  bash scripts/run_baseline.sh "$ROUNDS"
else
  echo
  echo "skipped baseline smoke test because RUN_BASELINE=$RUN_BASELINE"
fi

echo
echo "== ready =="
echo "baseline output: $ROOT_DIR/outputs/latest/output.zip"
echo "agent run: bash scripts/run_agent.sh 1 30 1"
