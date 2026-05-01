from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import time
from dataclasses import dataclass
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


SKILL_TEXT = """# Protein Ensemble Competition Skill

Use this skill when improving `solver.py` for AI4S task 3.

The only official inputs are `problems/1.json`, `problems/2.json`, and `problems/3.json`.
Each contains a single-chain amino-acid sequence and a reference conformer count.

Required output:
- mmCIF files named `{problem_id}_conf{N}_pred.cif`
- at most 10 conformers per problem
- finite single-chain coordinates with `_atom_site` and CA atoms
- `final_info.json` with diversity and validation-oriented metrics

Data governance:
- Do not use this competition's original MD trajectories, crystal structures, or NMR ensembles.
- Public pretrained models, public force fields, unrelated PDB entries, AlphaFold DB,
  UniProt/UniRef/MGnify, and unrelated public MD benchmark datasets are allowed, but
  optional. Never block a valid output on external downloads.

Iteration protocol:
1. Read `iteration_context.json`.
2. Write a concise `hypothesis.md` with at most 12 bullet lines.
3. If the hypothesis requires implementation, edit `solver.py`; observation-only iterations are allowed when justified.
4. Run a bounded smoke test before finishing when code changed.
5. Append concise evidence to `notes.md`.
"""


GOAL_TEMPLATE = """Iteration {iteration} of {total_iterations}: improve this workspace's solver.py for the AI4S protein ensemble competition.

Files available:
- solver.py: current sequence-only solver. You may edit it.
- problems/*.json: official problem inputs.
- README_AGENT.md: concise task spec.
- iteration_context.json: previous metrics, accepted solver history, and current constraints.
- .skills/protein-ensemble/SKILL.md: task-specific operating procedure.

Required behavior:
- CLI remains: python solver.py --problem-id 1 --sequence SEQ --num-conformers 4 --optimization-rounds {solver_rounds} --out-dir run
- Write valid mmCIF files into the requested out-dir.
- Include final_info.json with pairwise CA-RMSD, compactness, finite-coordinate, candidate-count, and optimization-round fields.
- Use no forbidden competition structures or trajectories.
- Keep runtime bounded.

This iteration must explicitly propose one hypothesis. Write it to `hypothesis.md` in at most 12 concise bullet lines.
Base changes on the prior observations in `iteration_context.json`; do not blindly rewrite the whole file.
You are allowed to leave `solver.py` unchanged when the iteration is observation-only, but say that explicitly in `hypothesis.md` and final answer.

If you edit `solver.py`, run `python print_sequence.py 1` to get the first sequence, then run:
python solver.py --problem-id 1 --sequence <that_sequence> --num-conformers 2 --optimization-rounds {solver_rounds} --out-dir smoke

Then summarize what changed.
"""


@dataclass
class IterationResult:
    iteration: int
    accepted: bool
    score: float
    solver_changed: bool
    report_path: str | None
    run_id: str | None
    stop_reason: str | None
    metrics: object
    events_path: str | None
    final_answer: str
    hypothesis: str
    error: str | None = None


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
    skill_dir = workspace / ".skills" / "protein-ensemble"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(SKILL_TEXT, encoding="utf-8")


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
        max_input_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_INPUT_TOKENS", "128000")),
        max_output_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_OUTPUT_TOKENS", "8192")),
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
        project_root=str(workspace),
        skills=("protein-ensemble",),
    )


def proxy_score(report: dict[str, object]) -> float:
    if not report.get("ok"):
        return -1.0
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return -1.0
    scores: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        info = result.get("final_info", {})
        if not isinstance(info, dict):
            continue
        diversity = float(info.get("pairwise_ca_rmsd_mean") or 0.0)
        generated = float(info.get("num_conformers_generated") or 0.0)
        requested = max(1.0, float(info.get("conformer_count") or generated or 1.0))
        finite = 1.0 if info.get("coordinate_finite") else 0.0
        candidate_count = float(info.get("candidate_count") or generated or 0.0)
        diversity_score = max(0.0, min(diversity, 25.0) / 25.0)
        count_score = max(0.0, min(generated / requested, 1.0))
        candidate_score = max(0.0, min(candidate_count / max(generated, 1.0), 3.0) / 3.0)
        scores.append(0.55 * diversity_score + 0.25 * count_score + 0.15 * finite + 0.05 * candidate_score)
    return round(sum(scores) / len(scores), 6) if scores else -1.0


def compact_report(report: dict[str, object]) -> dict[str, object]:
    return {
        "ok": report.get("ok"),
        "score_proxy": proxy_score(report),
        "results": [
            {
                "problem_id": item.get("problem_id"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "final_info": item.get("final_info"),
            }
            for item in report.get("results", [])
            if isinstance(item, dict)
        ],
    }


def write_iteration_context(
    workspace: Path,
    iteration: int,
    total_iterations: int,
    best_score: float,
    history: list[IterationResult],
) -> None:
    context = {
        "iteration": iteration,
        "total_iterations": total_iterations,
        "best_score_proxy": best_score,
        "accepted_history": [
            {
                "iteration": item.iteration,
                "score_proxy": item.score,
                "accepted": item.accepted,
                "solver_changed": item.solver_changed,
                "hypothesis": item.hypothesis[:2000],
                "error": item.error,
            }
            for item in history
        ],
        "score_proxy_definition": (
            "Internal selection proxy only: validation success, bounded CA diversity, "
            "requested conformer count coverage, finite coordinates, and candidate count. "
            "It is not the official hidden score."
        ),
        "code_change_policy": (
            "Code changes are encouraged but not mandatory. Observation-only iterations are valid "
            "when explicitly justified. The runner records solver_changed for audit."
        ),
    }
    (workspace / "iteration_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")


def load_hypothesis(workspace: Path) -> str:
    path = workspace / "hypothesis.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")[:6000]


def write_solver_diff(workspace: Path, iteration: int, before: str, after: str) -> bool:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"solver.before.iteration_{iteration:02d}.py",
            tofile=f"solver.after.iteration_{iteration:02d}.py",
        )
    )
    (workspace / f"solver_diff_{iteration:02d}.patch").write_text(diff, encoding="utf-8")
    return bool(diff)


def stopped_for_max_tokens(events_path: str | None) -> bool:
    if not events_path:
        return False
    path = Path(events_path)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "ASSISTANT_MESSAGE" and payload.get("stop_reason") == "max_tokens":
            return True
    return False


def run_agent_iterations(
    workspace: Path,
    iterations: int,
    solver_rounds: int,
    model: str,
    base_url: str | None,
) -> list[IterationResult]:
    history: list[IterationResult] = []
    best_solver = workspace / "best_solver.py"
    shutil.copy2(workspace / "solver.py", best_solver)
    best_score = -1.0
    iteration_root = workspace / "iteration_runs"
    iteration_root.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, iterations + 1):
        shutil.copy2(best_solver, workspace / "solver.py")
        solver_before = (workspace / "solver.py").read_text(encoding="utf-8")
        write_iteration_context(workspace, iteration, iterations, best_score, history)
        result = None
        report: dict[str, object] | None = None
        report_path: Path | None = None
        error: str | None = None
        solver_changed = False
        agent = build_agent(workspace, model, base_url)
        try:
            result = agent.run_sync(
                GOAL_TEMPLATE.format(
                    iteration=iteration,
                    total_iterations=iterations,
                    solver_rounds=solver_rounds,
                )
            )
            (workspace / f"agent_final_answer_{iteration:02d}.md").write_text(result.final_answer, encoding="utf-8")
            solver_after = (workspace / "solver.py").read_text(encoding="utf-8")
            solver_changed = write_solver_diff(workspace, iteration, solver_before, solver_after)
            if stopped_for_max_tokens(getattr(result, "events_path", None)):
                raise RuntimeError("agent response stopped at max_tokens before completing the iteration")
            report_path = iteration_root / f"iteration_{iteration:02d}"
            report = run_suite(workspace / "solver.py", report_path, solver_rounds, clean=True)
            score = proxy_score(report)
            accepted = bool(report.get("ok")) and score >= best_score
            if accepted:
                best_score = score
                shutil.copy2(workspace / "solver.py", best_solver)
            else:
                shutil.copy2(best_solver, workspace / "solver.py")
        except Exception as exc:
            score = -1.0
            accepted = False
            error = str(exc)
            shutil.copy2(best_solver, workspace / "solver.py")

        item = IterationResult(
            iteration=iteration,
            accepted=accepted,
            score=score,
            solver_changed=solver_changed,
            report_path=str(report_path) if report_path is not None else None,
            run_id=getattr(result, "run_id", None),
            stop_reason=getattr(result, "stop_reason", None),
            metrics=getattr(result, "metrics", None),
            events_path=getattr(result, "events_path", None),
            final_answer=(getattr(result, "final_answer", "") or "")[:4000],
            hypothesis=load_hypothesis(workspace),
            error=error,
        )
        history.append(item)
        (workspace / f"iteration_result_{iteration:02d}.json").write_text(
            json.dumps({
                **item.__dict__,
                "report": compact_report(report) if report is not None else None,
            }, indent=2),
            encoding="utf-8",
        )

    shutil.copy2(best_solver, workspace / "solver.py")
    (workspace / "iteration_summary.json").write_text(
        json.dumps([item.__dict__ for item in history], indent=2),
        encoding="utf-8",
    )
    return history


def append_iteration_audit(agent_log: Path, history: list[IterationResult]) -> None:
    for item in history:
        append_log(
            agent_log,
            "hypothesis_generation",
            iteration=item.iteration,
            hypothesis=item.hypothesis,
        )
        append_log(
            agent_log,
            "code_evolution",
            iteration=item.iteration,
            accepted=item.accepted,
            score_proxy=item.score,
            solver_changed=item.solver_changed,
            run_id=item.run_id,
            stop_reason=item.stop_reason,
            metrics=item.metrics,
            events_path=item.events_path,
            final_answer=item.final_answer,
            error=item.error,
        )
        append_log(
            agent_log,
            "experiment_observation",
            iteration=item.iteration,
            accepted=item.accepted,
            score_proxy=item.score,
            solver_changed=item.solver_changed,
            report_path=item.report_path,
            error=item.error,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--solver-rounds", type=int, default=1)
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

    history: list[IterationResult] = []
    if not args.skip_agent:
        history = run_agent_iterations(
            workspace=workspace,
            iterations=max(1, args.iterations),
            solver_rounds=max(1, args.solver_rounds),
            model=model,
            base_url=base_url,
        )

    out_root = Path(args.out)
    report = run_suite(workspace / "solver.py", out_root, max(1, args.solver_rounds), clean=True)
    if history:
        append_iteration_audit(out_root / "submission" / "agent.log", history)
        validation = validate_submission(out_root / "submission")
        package_output(out_root / "submission", out_root / "output.zip")
        report["validation"] = validation
        report["ok"] = validation["ok"]
        report["agent_iterations"] = [item.__dict__ for item in history]
        (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
