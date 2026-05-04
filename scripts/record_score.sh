#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
. "$ROOT_DIR/scripts/_python_env.sh"

"$PYTHON" -m protein_agent_tiny.score "$@"
