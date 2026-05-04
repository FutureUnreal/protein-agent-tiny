from __future__ import annotations

import shutil
from pathlib import Path

from ..prompts import load


_PRESERVE_ON_REPREP = {
    "solver_pkg",
    "best_pipeline",
    "research_plan.md",
    "hypothesis.md",
    "notes.md",
    "iteration_context.json",
    "baseline_result.json",
    "iteration_runs",
    "_bootstrap_smoke",
    "_emergency_submission",
    "solver.py",
}


def prepare_workspace(workspace: Path, root: Path) -> None:
    """Prepare a workspace idempotently for bootstrap/improve agents.

    When the workspace already contains agent-produced state (solver_pkg,
    best_pipeline, research artifacts), that state is preserved so that
    reruns can take the improve-agent path via solver_pkg/.pipeline_ready
    detection. Only regenerated scaffolding (problems, pyproject copy,
    skill doc, helper script, notes bootstrap) is refreshed.

    Does NOT copy any solver file. solver_pkg/ is created by the bootstrap
    agent on first run, and reused on subsequent runs via solver_pkg/.pipeline_ready.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    # Remove only regenerated scaffolding, not agent-produced state.
    for entry in workspace.iterdir():
        if entry.name in _PRESERVE_ON_REPREP:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                pass

    shutil.copy2(root / "pyproject.toml", workspace / "pyproject.toml")
    problems_dst = workspace / "problems"
    if problems_dst.exists():
        shutil.rmtree(problems_dst, ignore_errors=True)
    shutil.copytree(root / "data" / "problems", problems_dst)

    (workspace / "print_sequence.py").write_text(
        "import json, sys\n"
        "p = sys.argv[1]\n"
        "data = json.load(open(f'problems/{p}.json'))\n"
        "if isinstance(data, list):\n"
        "    data = data[0]\n"
        "seqs = data.get('sequences') or [{'proteinChain': data.get('proteinChain', {})}]\n"
        "print(seqs[0]['proteinChain']['sequence'])\n",
        encoding="utf-8",
    )

    skill_content = load("skill_protein_ensemble.md")
    (workspace / "README_AGENT.md").write_text(skill_content, encoding="utf-8")

    skill_dir = workspace / ".skills" / "protein-ensemble"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

    notes_path = workspace / "notes.md"
    if not notes_path.exists():
        notes_path.write_text("# Agent Notes\n\n", encoding="utf-8")
