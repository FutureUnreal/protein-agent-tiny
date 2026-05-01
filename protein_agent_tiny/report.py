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
            lines.extend([
                f"- Iteration `{item.get('iteration')}`: accepted=`{item.get('accepted')}`, score_proxy=`{item.get('score')}`, stop_reason=`{item.get('stop_reason')}`",
            ])
        lines.append("")
    lines.extend([
        "## Agent Audit",
        "",
        "`agent.log` is included in `output.zip` as JSONL. It records literature/data policy, approach decisions, hypothesis generation, code evolution, experiment runs, and validation observations.",
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
