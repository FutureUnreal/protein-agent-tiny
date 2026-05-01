from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .report import build_report
from .run_suite import DEFAULT_OUTPUT, ROOT, append_log, package_output, run_suite
from .validate import validate_submission


SYSTEM_PROMPT = """You are a small, competition-only AI4S protein ensemble agent.

Goal: improve solver.py for AI4S task 3 and keep the official submission path valid.

Hard constraints:
- Input is only amino-acid sequence for problems 1, 2, and 3.
- Do not use this competition's original MD trajectories, crystal structures, or NMR ensembles.
- You may optionally use public pretrained models, public force fields, RCSB PDB unrelated entries, AlphaFold DB, UniProt/UniRef/MGnify, or unrelated public MD benchmark datasets.
- Optional public resources must not block a valid output; keep a bounded sequence-only fallback.
- The solver must generate mmCIF files named {problem_id}_conf{N}_pred.cif.
- The final archive is output.zip and must contain CIF files plus agent.log at the zip root.
- Every loop must have strict limits. No unbounded rejection sampling or long training.

Engineering constraints:
- Edit only files in the current workspace.
- Keep solver.py runnable from CLI.
- Prefer compact, plausible, finite single-chain conformers with useful diversity.
- Record your reasoning and code changes in notes.md.
"""


GOAL_TEMPLATE = """Improve this workspace's solver.py for the AI4S protein ensemble competition.

Files available:
- solver.py: current sequence-only solver. You may edit it.
- problems/*.json: official problem inputs.
- README_AGENT.md: concise task spec.

Required behavior:
- CLI remains: python solver.py --problem-id 1 --sequence SEQ --num-conformers 4 --optimization-rounds 2 --out-dir run
- Write valid mmCIF files into the requested out-dir.
- Include final_info.json with pairwise CA-RMSD, compactness, finite-coordinate, candidate-count, and optimization-round fields.
- Use no forbidden competition structures or trajectories.
- Keep runtime bounded.

After edits, run `python print_sequence.py 1` to get the first sequence, then run:
python solver.py --problem-id 1 --sequence <that_sequence> --num-conformers 2 --optimization-rounds {rounds} --out-dir smoke

Then summarize what changed.
"""


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def prepare_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copy2(ROOT / "protein_agent_tiny" / "solver.py", workspace / "solver.py")
    shutil.copytree(ROOT / "data" / "problems", workspace / "problems")
    (workspace / "print_sequence.py").write_text(
        "import json, sys\n"
        "p=sys.argv[1]\n"
        "print(json.load(open(f'problems/{p}.json'))['proteinChain']['sequence'])\n",
        encoding="utf-8",
    )
    (workspace / "README_AGENT.md").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (workspace / "notes.md").write_text("# Agent Notes\n\n", encoding="utf-8")


def build_agent(workspace: Path, model: str, base_url: str | None):
    from all_in_agents import Agent, Budget, BUILTIN_TOOLS, OpenAIAdapter, ToolPolicy, ToolRegistry, unsafe_defaults

    from .tools import validate_submission_tool

    registry = ToolRegistry(approval_callback=unsafe_defaults())
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    registry.register(validate_submission_tool)
    budget = Budget(
        max_llm_calls=int(os.environ.get("PROTEIN_AGENT_MAX_LLM_CALLS", "20")),
        max_tool_calls=int(os.environ.get("PROTEIN_AGENT_MAX_TOOL_CALLS", "60")),
        max_wall_ms=int(os.environ.get("PROTEIN_AGENT_MAX_WALL_MS", "1800000")),
        loop_same_action_limit=4,
    )
    policy = ToolPolicy(
        require_approval_for=frozenset(),
        workspace_roots=(workspace.resolve(),),
        command_denylist=frozenset({"rm", "del", "rmdir"}),
        sanitize_env=False,
    )
    llm = OpenAIAdapter(model=model, base_url=base_url, max_retries=2)
    return Agent(
        llm=llm,
        tools=registry,
        budget=budget,
        run_dir=str(ROOT / "runs"),
        system=SYSTEM_PROMPT,
        workspace_root=str(workspace),
        tool_policy=policy,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    os.environ["PROTEIN_AGENT_MAX_WALL_MS"] = str(args.max_minutes * 60 * 1000)
    model = os.environ.get("PROTEIN_AGENT_MODEL") or os.environ.get("DEFAULT_MODEL") or "step-3.5-flash-2603"
    base_url = os.environ.get("OPENAI_API_BASE")
    workspace = Path(args.workspace) if args.workspace else ROOT / "workspaces" / time.strftime("%Y%m%d_%H%M%S")
    prepare_workspace(workspace)

    if not args.skip_agent:
        agent = build_agent(workspace, model, base_url)
        result = agent.run_sync(GOAL_TEMPLATE.format(rounds=args.rounds))
        (workspace / "agent_final_answer.md").write_text(result.final_answer, encoding="utf-8")
        (workspace / "agent_run.json").write_text(json.dumps({
            "run_id": result.run_id,
            "stop_reason": result.stop_reason,
            "metrics": result.metrics,
            "events_path": result.events_path,
        }, indent=2), encoding="utf-8")
    else:
        result = None

    out_root = Path(args.out)
    report = run_suite(workspace / "solver.py", out_root, args.rounds, clean=True)
    if result is not None:
        append_log(
            out_root / "submission" / "agent.log",
            "code_evolution",
            summary="all-in-agents completed a workspace edit run before final suite execution.",
            run_id=result.run_id,
            stop_reason=result.stop_reason,
            metrics=result.metrics,
            events_path=result.events_path,
            final_answer=result.final_answer[:4000],
        )
        validation = validate_submission(out_root / "submission")
        package_output(out_root / "submission", out_root / "output.zip")
        report["validation"] = validation
        report["ok"] = validation["ok"]
        (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
