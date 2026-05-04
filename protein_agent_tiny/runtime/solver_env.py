"""Resolve the Python interpreter that runs solver_pkg/cli.py.

The agent runtime always runs in the project's uv-managed .venv. But the
solver subprocess can run in a different interpreter — typically a Docker
image that already has torch / esmfold / biotite installed system-wide.
Reusing the system interpreter avoids duplicating multi-GB ML dependencies
in the project venv.

Resolution order:
  1. $PROTEIN_AGENT_SOLVER_PYTHON — explicit override.
  2. Probe candidate system interpreters (`python3`, `python`) for the
     scientific-stack import set; the first one that imports useful packages
     wins. "Useful" = at least numpy AND one of {torch, biotite, esm,
     transformers}. A bare numpy is not enough to prefer system over .venv.
  3. Fall back to sys.executable (the agent runtime's own .venv python).
     The agent is then expected to add deps to pyproject.toml when needed.

The choice is recorded in environment_report so the bootstrap/improve agent
can see which packages are already available where.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# Packages whose presence indicates a "scientific" host environment that's
# better than the project's own .venv. numpy alone is not enough — the project
# venv has it too. We want to detect torch / biotite / esm / transformers etc.
HOST_INDICATOR_PACKAGES = ("torch", "biotite", "esm", "transformers", "openmm", "Bio")


@dataclass(frozen=True)
class SolverEnvProbe:
    python: str
    source: str  # "override" | "host" | "venv"
    available_packages: dict  # package_name -> bool
    notes: str


def _probe_imports(python: str, packages: tuple) -> dict:
    """Return {pkg: bool} for whether `python -c 'import pkg'` succeeds."""
    if not python:
        return {p: False for p in packages}
    code = (
        "import json, importlib.util as u\n"
        "pkgs = " + json.dumps(list(packages)) + "\n"
        "print(json.dumps({p: u.find_spec(p) is not None for p in pkgs}))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return {p: False for p in packages}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {p: False for p in packages}


def _is_host_useful(probe: dict) -> bool:
    """A host interpreter is preferred over .venv only if it offers more than numpy."""
    return any(probe.get(p) for p in ("torch", "biotite", "esm", "transformers", "openmm", "Bio"))


def resolve_solver_python(packages: tuple = HOST_INDICATOR_PACKAGES) -> SolverEnvProbe:
    override = os.environ.get("PROTEIN_AGENT_SOLVER_PYTHON")
    if override and Path(override).exists():
        return SolverEnvProbe(
            python=override,
            source="override",
            available_packages=_probe_imports(override, packages + ("numpy",)),
            notes="Selected via PROTEIN_AGENT_SOLVER_PYTHON env var.",
        )

    # Try host interpreters that are NOT the agent runtime's sys.executable.
    runtime_py = Path(sys.executable).resolve()
    for candidate_name in ("python3", "python"):
        candidate = shutil.which(candidate_name)
        if not candidate:
            continue
        if Path(candidate).resolve() == runtime_py:
            continue  # same as agent runtime; not a host environment.
        host_probe = _probe_imports(candidate, packages + ("numpy",))
        if _is_host_useful(host_probe):
            return SolverEnvProbe(
                python=candidate,
                source="host",
                available_packages=host_probe,
                notes=(
                    "Detected scientific stack on system Python; reusing it for solver "
                    "to avoid duplicating heavy ML deps inside .venv."
                ),
            )

    # Fall back to agent runtime's own python.
    return SolverEnvProbe(
        python=sys.executable,
        source="venv",
        available_packages=_probe_imports(sys.executable, packages + ("numpy",)),
        notes=(
            "No suitable host interpreter found; using project .venv python. "
            "Agent should add scientific dependencies to pyproject.toml when needed."
        ),
    )
