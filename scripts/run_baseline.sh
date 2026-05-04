#!/usr/bin/env bash
# Package-only audit smoke test. NOT a scientific submission.
#
# Runs `agent_runner --skip-agent`, which bypasses all LLM agents and writes
# ZERO CIF files into the submission directory, plus a complete agent.log marked
# `audit_only=true, mode=package_only`. Use this only to verify packaging,
# validation plumbing, and CI wiring. The resulting output.zip is intentionally
# empty of science and MUST NOT be submitted as a competition entry.
set -euo pipefail

rounds="${1:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
. "$ROOT_DIR/scripts/_python_env.sh"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

"$PYTHON" -m protein_agent_tiny.agent_runner \
  --skip-agent \
  --iterations 1 \
  --solver-rounds "$rounds" \
  --out outputs/latest

echo "Package-only audit run completed."
echo "Output: $ROOT_DIR/outputs/latest/output.zip (audit_only, contains zero CIFs)"
echo "WARNING: this output.zip is NOT a valid competition submission."
