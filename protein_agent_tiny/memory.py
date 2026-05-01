from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def memory_dir(root: Path) -> Path:
    path = root / "memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-limit:]


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_memory(root: Path) -> dict[str, Any]:
    mem = memory_dir(root)
    observations_path = mem / "observations.md"
    return {
        "observations": observations_path.read_text(encoding="utf-8") if observations_path.exists() else "",
        "recent_runs": read_jsonl(mem / "runs.jsonl", limit=12),
        "best_runs": read_jsonl(mem / "best_runs.jsonl", limit=8),
    }


def render_memory_context(memory: dict[str, Any]) -> str:
    lines = ["# Long-Term Memory", ""]
    observations = str(memory.get("observations") or "").strip()
    if observations:
        lines.extend(["## Factual Observations", "", observations[-6000:], ""])
    else:
        lines.extend(["No prior observations recorded.", ""])
    best_runs = memory.get("best_runs") or []
    if best_runs:
        lines.extend(["## Best Runs", ""])
        for run in best_runs[-5:]:
            lines.append(
                f"- `{run.get('timestamp_unix')}` score=`{run.get('best_score')}` "
                f"workspace=`{run.get('workspace')}` notes=`{run.get('summary')}`"
            )
        lines.append("")
    recent_runs = memory.get("recent_runs") or []
    if recent_runs:
        lines.extend(["## Recent Runs", ""])
        for run in recent_runs[-5:]:
            lines.append(
                f"- `{run.get('timestamp_unix')}` iterations=`{run.get('iterations')}` "
                f"best_score=`{run.get('best_score')}` accepted=`{run.get('accepted_count')}`"
            )
        lines.append("")
    return "\n".join(lines)


def write_memory_context(root: Path, workspace: Path) -> dict[str, Any]:
    data = load_memory(root)
    (workspace / "memory_context.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (workspace / "memory_context.md").write_text(render_memory_context(data), encoding="utf-8")
    return data


def summarize_run(history: list[Any], final_report: dict[str, Any], workspace: Path) -> dict[str, Any]:
    accepted = [item for item in history if getattr(item, "accepted", False)]
    best_score = max((float(getattr(item, "score", -1.0)) for item in accepted), default=-1.0)
    best_iteration = None
    for item in accepted:
        if float(getattr(item, "score", -1.0)) == best_score:
            best_iteration = getattr(item, "iteration", None)
    result_metrics = []
    for result in final_report.get("results", []):
        if not isinstance(result, dict):
            continue
        info = result.get("final_info", {})
        if isinstance(info, dict):
            result_metrics.append({
                "problem_id": result.get("problem_id"),
                "pairwise_ca_rmsd_mean": info.get("pairwise_ca_rmsd_mean"),
                "radius_of_gyration_mean": info.get("radius_of_gyration_mean"),
                "candidate_count": info.get("candidate_count"),
            })
    return {
        "timestamp_unix": int(time.time()),
        "workspace": str(workspace),
        "iterations": len(history),
        "accepted_count": len(accepted),
        "best_score": best_score,
        "best_iteration": best_iteration,
        "output_zip": final_report.get("output_zip"),
        "validation_ok": final_report.get("ok"),
        "result_metrics": result_metrics,
        "summary": "Factual record from bounded hypothesis-experiment-reflection run.",
    }


def update_memory(
    root: Path,
    workspace: Path,
    history: list[Any],
    final_report: dict[str, Any],
    literature: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    mem = memory_dir(root)
    run_summary = summarize_run(history, final_report, workspace)
    run_summary["literature_paper_count"] = literature.get("paper_count")
    run_summary["environment"] = {
        "cpu_count": environment.get("cpu_count"),
        "disk_free_gb": (environment.get("disk") or {}).get("free_gb"),
        "torch_available": (environment.get("python_modules") or {}).get("torch"),
        "nvidia_smi": ((environment.get("commands") or {}).get("nvidia_smi") or {}).get("available"),
    }
    append_jsonl(mem / "runs.jsonl", run_summary)
    if run_summary["best_score"] >= 0:
        append_jsonl(mem / "best_runs.jsonl", run_summary)

    observations = [
        f"## Run {run_summary['timestamp_unix']}",
        "",
        f"- Workspace: `{workspace}`",
        f"- Iterations: `{run_summary['iterations']}`, accepted: `{run_summary['accepted_count']}`, best score: `{run_summary['best_score']}`",
        f"- Literature papers retrieved: `{literature.get('paper_count')}`",
        f"- Environment: cpu_count=`{run_summary['environment'].get('cpu_count')}`, "
        f"torch_available=`{run_summary['environment'].get('torch_available')}`, "
        f"nvidia_smi=`{run_summary['environment'].get('nvidia_smi')}`",
    ]
    for metric in run_summary["result_metrics"]:
        observations.append(
            f"- Problem `{metric.get('problem_id')}` result: pairwise_CA_RMSD_mean="
            f"`{metric.get('pairwise_ca_rmsd_mean')}`, Rg_mean="
            f"`{metric.get('radius_of_gyration_mean')}`, candidates=`{metric.get('candidate_count')}`"
        )
    for item in history[-5:]:
        observations.append(
            f"- Iteration `{getattr(item, 'iteration', None)}` accepted=`{getattr(item, 'accepted', None)}` "
            f"score=`{getattr(item, 'score', None)}` changed=`{getattr(item, 'solver_changed', None)}` "
            f"dependency_changed=`{getattr(item, 'dependency_changed', None)}` "
            f"error=`{getattr(item, 'error', None)}`"
        )
        observation = str(getattr(item, "observation", "") or "").strip().replace("\n", " ")
        if observation:
            observations.append(f"  Recorded observation: {observation[:600]}")
    observations.append("")
    with (mem / "observations.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(observations) + "\n")

    shutil.copy2(workspace / "environment_report.json", mem / "environment_report.json")
    return run_summary
