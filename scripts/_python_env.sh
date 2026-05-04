#!/usr/bin/env bash
# Shared python interpreter resolver.
#
# Resolution order (first match wins):
#   1. $PROTEIN_AGENT_PYTHON — explicit override.
#   2. Project .venv/ (POSIX or Windows layout). Preferred because it isolates
#      the project's dependencies (e.g. the uv-managed deps in pyproject.toml).
#   3. System `python` / `python3` — but only if it can `import protein_agent_tiny`.
#      This is the "Docker image / pre-baked environment" path: if the image
#      already has deps installed globally, skip building another venv.
#   4. Fail with guidance.
#
# After sourcing this script, $PYTHON is set and exported.
#
# Usage:
#   # shellcheck disable=SC1091
#   . "$(dirname "${BASH_SOURCE[0]}")/_python_env.sh"

resolve_python() {
  local root_dir="$1"
  local candidate=""

  # 1. Explicit override.
  if [ -n "${PROTEIN_AGENT_PYTHON:-}" ] && [ -x "$PROTEIN_AGENT_PYTHON" ]; then
    candidate="$PROTEIN_AGENT_PYTHON"
    echo "$candidate"
    return 0
  fi

  # 2. Project venv — POSIX layout then Windows layout.
  if [ -x "$root_dir/.venv/bin/python" ]; then
    candidate="$root_dir/.venv/bin/python"
    echo "$candidate"
    return 0
  fi
  if [ -f "$root_dir/.venv/Scripts/python.exe" ]; then
    candidate="$root_dir/.venv/Scripts/python.exe"
    echo "$candidate"
    return 0
  fi

  # 3. System python, but only if the package is importable.
  for sys_py in python python3; do
    if command -v "$sys_py" >/dev/null 2>&1; then
      if (cd "$root_dir" && "$sys_py" -c "import protein_agent_tiny" >/dev/null 2>&1); then
        candidate="$(command -v "$sys_py")"
        echo "$candidate"
        return 0
      fi
    fi
  done

  # 4. No working interpreter.
  return 1
}

__PAT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! PYTHON="$(resolve_python "$__PAT_ROOT_DIR")"; then
  cat >&2 <<EOF
Error: no usable Python found.

Looked for (in order):
  1. \$PROTEIN_AGENT_PYTHON                    (not set or not executable)
  2. $__PAT_ROOT_DIR/.venv/bin/python          (missing)
  3. $__PAT_ROOT_DIR/.venv/Scripts/python.exe  (missing)
  4. system python / python3 that can        (neither can import
     \`import protein_agent_tiny\`              protein_agent_tiny)

Fix one of:
  - Local dev:   scripts/deploy_uv.sh         (creates .venv via uv)
  - Docker/CI:   pip install -e .             (install project into system env)
  - Custom:      export PROTEIN_AGENT_PYTHON=/path/to/python
EOF
  exit 1
fi
export PYTHON
