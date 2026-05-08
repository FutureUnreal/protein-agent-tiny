"""Resolve the Python interpreter that runs solver_pkg/cli.py.

The framework runtime runs in the project root uv environment. The generated
solver is allowed to use a separate environment owned by the current workspace.

Resolution order:
  1. PROTEIN_AGENT_SOLVER_PYTHON, when explicitly set.
  2. Local system Python with torch already installed.
  3. Workspace .venv created from workspaces/current/pyproject.toml via uv.
  4. Fail loudly. The project runtime Python is not a solver fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


HOST_INDICATOR_PACKAGES = ("torch", "biotite", "esm", "transformers", "openmm", "Bio")


@dataclass(frozen=True)
class SolverEnvProbe:
    python: str
    source: str  # "override" | "host" | "workspace"
    available_packages: dict
    notes: str


def _probe_imports(python: str, packages: tuple) -> dict:
    if not python:
        return {p: False for p in packages}
    code = (
        "import json, importlib.util as u\n"
        "pkgs = " + json.dumps(list(packages)) + "\n"
        "print(json.dumps({p: u.find_spec(p) is not None for p in pkgs}))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return {p: False for p in packages}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {p: False for p in packages}


def _workspace_python(workspace: Path) -> Path:
    if os.name == "nt":
        return workspace / ".venv" / "Scripts" / "python.exe"
    return workspace / ".venv" / "bin" / "python"


def _candidate_system_pythons() -> list[str]:
    candidates: list[str] = []
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for path in (
        "/usr/bin/python3",
        "/usr/bin/python",
        "/usr/local/bin/python3",
        "/usr/local/bin/python",
    ):
        if Path(path).exists():
            candidates.append(path)

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path.home() / ".local" / "bin" / "uv.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _install_uv() -> str | None:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    else:
        command = ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    return _find_uv()


def _ensure_workspace_pyproject(workspace: Path) -> None:
    pyproject = workspace / "pyproject.toml"
    if pyproject.exists():
        return
    pyproject.write_text(
        "[project]\n"
        'name = "protein-agent-workspace-solver"\n'
        'version = "0.0.0"\n'
        'requires-python = ">=3.10"\n'
        "dependencies = [\n"
        '  "numpy>=1.26",\n'
        '  "torch>=2.2.0,<2.7",\n'
        "]\n"
        "\n"
        "[tool.uv]\n"
        "package = false\n",
        encoding="utf-8",
    )


def _ensure_workspace_env(workspace: Path) -> tuple[str | None, str]:
    workspace.mkdir(parents=True, exist_ok=True)
    _ensure_workspace_pyproject(workspace)

    workspace_python = _workspace_python(workspace)
    if workspace_python.exists():
        probe = _probe_imports(str(workspace_python), ("torch", "numpy"))
        if probe.get("torch") and probe.get("numpy"):
            return str(workspace_python), "Reusing existing workspace .venv."

    uv = _find_uv() or _install_uv()
    if not uv:
        return None, "uv is unavailable and automatic uv installation failed."

    proc = subprocess.run(
        [uv, "sync"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        return None, f"uv sync failed: {proc.stderr.strip()[:1000]}"
    if workspace_python.exists():
        probe = _probe_imports(str(workspace_python), ("torch", "numpy"))
        if not (probe.get("torch") and probe.get("numpy")):
            return None, f"uv sync completed but torch/numpy are still unavailable: {probe}"
        return str(workspace_python), "Created workspace .venv with uv sync."
    return None, "uv sync completed but workspace Python was not found."


def resolve_solver_python(
    workspace: Path | None = None,
    packages: tuple = HOST_INDICATOR_PACKAGES,
) -> SolverEnvProbe:
    override = os.environ.get("PROTEIN_AGENT_SOLVER_PYTHON")
    if override:
        if not Path(override).exists():
            raise RuntimeError(f"PROTEIN_AGENT_SOLVER_PYTHON does not exist: {override}")
        return SolverEnvProbe(
            python=override,
            source="override",
            available_packages=_probe_imports(override, packages + ("numpy",)),
            notes="Selected via PROTEIN_AGENT_SOLVER_PYTHON.",
        )

    runtime_py = Path(sys.executable).resolve()
    for candidate in _candidate_system_pythons():
        if Path(candidate).resolve() == runtime_py:
            continue
        probe = _probe_imports(candidate, packages + ("numpy",))
        if probe.get("torch"):
            return SolverEnvProbe(
                python=candidate,
                source="host",
                available_packages=probe,
                notes="Detected torch on local system Python; using it directly.",
            )

    if workspace is not None:
        workspace_python, note = _ensure_workspace_env(workspace)
        if workspace_python:
            return SolverEnvProbe(
                python=workspace_python,
                source="workspace",
                available_packages=_probe_imports(workspace_python, packages + ("numpy",)),
                notes=f"{note} Controlled by the workspace pyproject.toml.",
            )

    raise RuntimeError(
        "No usable solver Python found. Set PROTEIN_AGENT_SOLVER_PYTHON, install torch "
        "into a local system Python, or provide a workspace where uv can create .venv."
    )
