#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run scripts/deploy_uv.sh first." >&2
  exit 1
fi

.venv/bin/python -m protein_agent_tiny.run_suite --clean --rounds "${1:-1}"
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
.venv/bin/python -m protein_agent_tiny.report --run-dir outputs/latest
ARCHIVE_DIR="$(
  .venv/bin/python -m protein_agent_tiny.archive --run-dir outputs/latest --label baseline
)"

echo "output.zip: $ROOT_DIR/outputs/latest/output.zip"
echo "archive: $ARCHIVE_DIR"
