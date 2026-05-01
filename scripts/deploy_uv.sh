#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --python 3.10

if [ ! -f .env ]; then
  cp .env.example .env
fi

mkdir -p outputs workspaces runs
echo "protein-agent-tiny ready"
echo "python: $ROOT_DIR/.venv/bin/python"
