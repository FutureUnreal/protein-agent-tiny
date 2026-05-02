from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(run_dir: Path) -> str:
    data = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    lines = [
        "# AI4S Protein Ensemble Tiny Agent Report",
        "",
        "## Summary",
        "",
        f"- Submission archive: `{data['output_zip']}`",
        f"- Archived snapshot: `{data.get('archive_dir')}`",
        f"- Validation: `{data['ok']}`",
        "- Data policy: sequence input only for competition problems; no competition MD trajectory, crystal structure, or NMR ensemble was used.",
        "- Optional allowed public resources are reserved for future agent improvements: RCSB PDB, AlphaFold DB, UniProt/UniRef/MGnify, public unrelated MD benchmarks.",
        "",
    ]
    literature = data.get("literature") or {}
    if literature:
        lines.extend([
            "## Literature",
            "",
            f"- Source: `{literature.get('source')}`",
            f"- Retrieved papers: `{literature.get('paper_count')}`",
            f"- Queries: `{', '.join(literature.get('queries') or [])}`",
            "",
        ])
    environment = data.get("environment") or {}
    if environment:
        commands = environment.get("commands") or {}
        modules = environment.get("python_modules") or {}
        lines.extend([
            "## Environment",
            "",
            f"- CPU count: `{environment.get('cpu_count')}`",
            f"- Disk free GB: `{(environment.get('disk') or {}).get('free_gb')}`",
            f"- NVIDIA SMI available: `{commands.get('nvidia_smi')}`",
            f"- Torch available: `{modules.get('torch')}`",
            "",
        ])
    memory = data.get("memory_summary") or {}
    if memory:
        lines.extend([
            "## Memory",
            "",
            f"- Best score recorded: `{memory.get('best_score')}`",
            f"- Accepted iterations: `{memory.get('accepted_count')}`",
            f"- Memory workspace: `{memory.get('workspace')}`",
            "",
        ])
    lines.extend([
        "## Results",
        "",
    ])
    for result in data["results"]:
        info = result.get("final_info", {})
        lines.extend([
            f"### Problem {result['problem_id']}",
            "",
            f"- Sequence length: `{info.get('sequence_length')}`",
            f"- Conformers: `{info.get('num_conformers_generated')}`",
            f"- Pairwise CA-RMSD mean: `{info.get('pairwise_ca_rmsd_mean')}`",
            f"- Radius of gyration mean: `{info.get('radius_of_gyration_mean')}`",
            f"- Runtime seconds: `{info.get('runtime_seconds')}`",
            "",
        ])
    iterations = data.get("agent_iterations") or []
    if iterations:
        lines.extend([
            "## Agent Iterations",
            "",
        ])
        for item in iterations:
            hard_gate = ((item.get("report") or {}).get("hard_gate_violations") or []) if isinstance(item.get("report"), dict) else []
            lines.extend([
                f"- Iteration `{item.get('iteration')}`: accepted=`{item.get('accepted')}`, score_proxy=`{item.get('score')}`, solver_changed=`{item.get('solver_changed')}`, dependency_changed=`{item.get('dependency_changed')}`, stop_reason=`{item.get('stop_reason')}`, hard_gate_violations=`{hard_gate}`",
            ])
        lines.append("")
    lines.extend([
        "## Agent Audit",
        "",
        "`agent.log` is included in `output.zip` as JSONL. It records literature/data policy, environment probing, research planning, hypothesis generation, code and dependency evolution, experiment runs, factual observations, and validation results.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/latest")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    text = build_report(run_dir)
    output = run_dir / "technical_report.md"
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
