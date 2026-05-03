from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .archive import archive_output
from .environment import write_environment_report
from .literature import collect_literature
from .memory import update_memory, write_memory_context
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
- `pyproject.toml` is the only dependency manifest. You may edit it and install
  public Python dependencies when justified by the environment report, but solver.py
  must keep a bounded fallback if an optional dependency is unavailable.
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
2. Read `memory_context.md`, `environment_report.md`, and `literature_review.md`.
3. Write `research_plan.md` first. Choose the iteration mode yourself: literature review,
   environment setup, dependency experiment, modeling, scoring analysis, code evolution,
   or observation-only audit.
4. Cite one relevant prior fact, environment constraint, or literature implication in `hypothesis.md`.
5. Write a concise `hypothesis.md` with at most 12 bullet lines.
6. If the hypothesis requires implementation, edit `solver.py`; observation-only iterations are allowed when justified.
7. Run a bounded smoke test before finishing when code changed or dependencies changed.
8. Append concise evidence to `notes.md`.
"""


GOAL_TEMPLATE = """Iteration {iteration} of {total_iterations}: improve this workspace's solver.py for the AI4S protein ensemble competition.

Files available:
- solver.py: current sequence-only solver. You may edit it.
- pyproject.toml: the only dependency manifest. You may edit it for dependency planning.
- problems/*.json: official problem inputs.
- README_AGENT.md: concise task spec.
- literature_review.md and literature_sources.json: OpenAlex literature retrieval results for architecture inspiration.
- environment_report.md/json: CPU/GPU/memory/package availability for choosing bounded methods.
- memory_context.md/json: prior run lessons and best run summaries.
- iteration_context.json: previous metrics, accepted solver history, and current constraints.
- .skills/protein-ensemble/SKILL.md: task-specific operating procedure.

Required behavior:
- CLI remains: python solver.py --problem-id 1 --sequence SEQ --num-conformers 4 --optimization-rounds {solver_rounds} --out-dir run
- Write valid mmCIF files into the requested out-dir.
- Include final_info.json with pairwise CA-RMSD, compactness, finite-coordinate, candidate-count, and optimization-round fields.
- Use no forbidden competition structures or trajectories.
- Keep runtime bounded.
- Optional dependencies are allowed only when public, appropriate for this environment, and not required for a valid fallback.
- If you edit pyproject.toml, record the dependency rationale in notes.md and run a bounded import/smoke test.
- Keep final responses concise. Do not paste long code, long command output, or full JSON into the final answer.

First write `research_plan.md`. It must state:
- selected mode for this iteration
- facts considered from memory, environment, literature, and prior results
- candidate actions considered
- chosen action and bounded validation plan

Then explicitly propose one hypothesis. Write it to `hypothesis.md` in at most 12 concise bullet lines.
Base changes on the prior observations in `iteration_context.json`, available compute in `environment_report.md`, factual observations in `memory_context.md`, and at least one relevant point from `literature_review.md`; do not blindly rewrite the whole file.
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
    dependency_changed: bool
    report_path: str | None
    run_id: str | None
    stop_reason: str | None
    metrics: object
    events_path: str | None
    final_answer: str
    artifact_validation: object
    research_plan: str
    hypothesis: str
    observation: str
    reflection_run_id: str | None = None
    reflection_stop_reason: str | None = None
    reflection_metrics: object = None
    reflection_events_path: str | None = None
    warnings: list[str] | None = None
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
    shutil.copy2(ROOT / "pyproject.toml", workspace / "pyproject.toml")
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
    from all_in_agents import Agent, ArtifactContract, ArtifactSpec, Budget, BUILTIN_TOOLS, OpenAIAdapter, ToolPolicy, ToolRegistry, unsafe_defaults

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
    artifact_contract = ArtifactContract((
        ArtifactSpec("research_plan.md", min_bytes=200, description="Iteration research plan."),
        ArtifactSpec("hypothesis.md", min_bytes=80, description="Concise iteration hypothesis."),
        ArtifactSpec("notes.md", min_bytes=20, description="Evidence and smoke-test notes."),
        ArtifactSpec("solver.py", min_bytes=1000, description="Runnable solver artifact."),
    ))
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
        artifact_contract=artifact_contract,
    )


def build_reflection_agent(workspace: Path, model: str, base_url: str | None):
    from all_in_agents import Agent, Budget, OpenAIAdapter, ToolPolicy, ToolRegistry

    budget = Budget(
        max_llm_calls=1,
        max_tool_calls=0,
        max_wall_ms=180000,
        max_input_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_INPUT_TOKENS", "128000")),
        max_output_tokens_per_call=min(int(os.environ.get("PROTEIN_AGENT_MAX_OUTPUT_TOKENS", "8192")), 4096),
    )
    policy = ToolPolicy(
        require_approval_for=frozenset(),
        workspace_roots=(workspace.resolve(),),
        command_denylist=frozenset({"rm", "del", "rmdir"}),
        sanitize_env=False,
    )
    system = (
        "You are the reflection phase of a protein ensemble research agent. "
        "You cannot edit files or run tools. Analyze the completed experiment, "
        "explain what changed, whether the hypothesis was supported, and record unresolved factual questions."
    )
    return Agent(
        llm=OpenAIAdapter(model=model, base_url=base_url, max_retries=2),
        tools=ToolRegistry(),
        budget=budget,
        run_dir=str(ROOT / "runs"),
        system=system,
        workspace_root=str(workspace),
        tool_policy=policy,
        project_root=str(workspace),
        skills=("protein-ensemble",),
    )


def bounded_peak_score(value: float, low: float, target: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value <= target:
        return min(1.0, (value - low) / max(target - low, 1e-6))
    if value >= high:
        return 0.0
    return max(0.0, 1.0 - (value - target) / max(high - target, 1e-6))


def rg_plausibility_score(rg: float, sequence_length: float) -> float:
    if rg <= 0 or sequence_length <= 0:
        return 0.0
    upper = 2.8 * (sequence_length ** 0.52)
    if rg <= upper:
        return 1.0
    return max(0.0, 1.0 - (rg - upper) / max(1.2 * upper, 1e-6))


def ca_spacing_score(min_distance: float, max_distance: float) -> float:
    if min_distance <= 0 or max_distance <= 0 or min_distance > max_distance:
        return 0.0
    low_error = max(0.0, 3.65 - min_distance)
    high_error = max(0.0, max_distance - 3.95)
    return max(0.0, 1.0 - (low_error + high_error) / 0.8)


def hard_gate_violations(report: dict[str, object]) -> list[str]:
    violations: list[str] = []
    if not report.get("ok"):
        violations.append("suite_not_ok")
    results = report.get("results", [])
    if not isinstance(results, list) or not results:
        return violations + ["missing_results"]
    for result in results:
        if not isinstance(result, dict):
            violations.append("malformed_result")
            continue
        problem_id = result.get("problem_id")
        info = result.get("final_info", {})
        if not isinstance(info, dict):
            violations.append(f"{problem_id}:missing_final_info")
            continue
        sequence_length = float(info.get("sequence_length") or 0.0)
        generated = float(info.get("num_conformers_generated") or 0.0)
        rg = float(info.get("radius_of_gyration_mean") or 0.0)
        diversity = float(info.get("pairwise_ca_rmsd_mean") or 0.0)
        min_ca = float(info.get("min_ca_distance") or 0.0)
        max_ca = float(info.get("max_ca_distance") or 0.0)
        rg_limit = max(220.0, 8.0 * (sequence_length ** 0.52)) if sequence_length > 0 else 220.0
        rmsd_limit = max(120.0, 4.0 * rg_limit)
        if generated <= 0:
            violations.append(f"{problem_id}:no_conformers")
        if not info.get("coordinate_finite"):
            violations.append(f"{problem_id}:nonfinite_coordinates")
        if rg <= 0 or rg > rg_limit:
            violations.append(f"{problem_id}:implausible_rg:{rg:.3f}>{rg_limit:.3f}")
        if diversity < 0 or diversity > rmsd_limit:
            violations.append(f"{problem_id}:implausible_pairwise_rmsd:{diversity:.3f}>{rmsd_limit:.3f}")
        if min_ca and min_ca < 2.7:
            violations.append(f"{problem_id}:ca_spacing_too_short:{min_ca:.3f}")
        if max_ca and max_ca > 5.2:
            violations.append(f"{problem_id}:ca_spacing_too_long:{max_ca:.3f}")
    return violations


def proxy_score(report: dict[str, object]) -> float:
    if hard_gate_violations(report):
        return -1.0
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
        sequence_length = float(info.get("sequence_length") or 0.0)
        diversity = float(info.get("pairwise_ca_rmsd_mean") or 0.0)
        rg = float(info.get("radius_of_gyration_mean") or 0.0)
        min_ca = float(info.get("min_ca_distance") or 0.0)
        max_ca = float(info.get("max_ca_distance") or 0.0)
        generated = float(info.get("num_conformers_generated") or 0.0)
        requested = max(1.0, float(info.get("conformer_count") or generated or 1.0))
        finite = 1.0 if info.get("coordinate_finite") else 0.0
        candidate_count = float(info.get("candidate_count") or generated or 0.0)
        rg_upper = 2.8 * (sequence_length ** 0.52) if sequence_length > 0 else 25.0
        diversity_target = max(6.0, min(25.0, 0.18 * rg_upper))
        diversity_score = bounded_peak_score(diversity, 1.5, diversity_target, diversity_target * 3.0)
        rg_score = rg_plausibility_score(rg, sequence_length)
        spacing_score = ca_spacing_score(min_ca, max_ca)
        count_score = max(0.0, min(generated / requested, 1.0))
        candidate_score = max(0.0, min(candidate_count / max(generated, 1.0), 3.0) / 3.0)
        scores.append(
            0.30 * diversity_score
            + 0.20 * count_score
            + 0.20 * rg_score
            + 0.15 * finite
            + 0.10 * spacing_score
            + 0.05 * candidate_score
        )
    return round(sum(scores) / len(scores), 6) if scores else -1.0


def compact_report(report: dict[str, object]) -> dict[str, object]:
    return {
        "ok": report.get("ok"),
        "score_proxy": proxy_score(report),
        "hard_gate_violations": hard_gate_violations(report),
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
        "baseline_result_path": "baseline_result.json",
        "accepted_history": [
            {
                "iteration": item.iteration,
                "score_proxy": item.score,
                "accepted": item.accepted,
                "solver_changed": item.solver_changed,
                "dependency_changed": item.dependency_changed,
                "research_plan": item.research_plan[:2000],
                "hypothesis": item.hypothesis[:2000],
                "observation": item.observation[:2000],
                "error": item.error,
            }
            for item in history
        ],
        "literature_review_path": "literature_review.md",
        "literature_sources_path": "literature_sources.json",
        "environment_report_path": "environment_report.md",
        "memory_context_path": "memory_context.md",
        "score_proxy_definition": (
            "Internal selection proxy only: validation success, bounded CA diversity, "
            "plausible radius of gyration, CA spacing, requested conformer count coverage, "
            "finite coordinates, and candidate count. "
            "It is not the official hidden score."
        ),
        "hard_gate_definition": (
            "Experiments with missing output, nonfinite coordinates, no conformers, extreme radius "
            "of gyration, extreme pairwise RMSD, or gross CA spacing violations are rejected before "
            "proxy scoring."
        ),
        "code_change_policy": (
            "Code changes are encouraged but not mandatory. Observation-only iterations are valid "
            "when explicitly justified in research_plan.md and hypothesis.md. Missing files are rejected."
        ),
        "dependency_policy": (
            "The workspace pyproject.toml is the only dependency manifest. The agent may edit it "
            "for public dependencies justified by environment_report.md, and must run a bounded "
            "import/smoke test when dependencies change. solver.py must keep a valid fallback if "
            "optional imports fail."
        ),
    }
    (workspace / "iteration_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")


def load_research_plan(workspace: Path) -> str:
    path = workspace / "research_plan.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")[:6000]


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


def write_dependency_diff(
    workspace: Path,
    iteration: int,
    before: dict[str, str],
    after: dict[str, str],
) -> bool:
    parts: list[str] = []
    for name in sorted(set(before) | set(after)):
        if before.get(name, "") == after.get(name, ""):
            continue
        parts.append(
            "".join(
                difflib.unified_diff(
                    before.get(name, "").splitlines(keepends=True),
                    after.get(name, "").splitlines(keepends=True),
                    fromfile=f"{name}.before.iteration_{iteration:02d}",
                    tofile=f"{name}.after.iteration_{iteration:02d}",
                )
            )
        )
    diff = "".join(parts)
    (workspace / f"dependency_diff_{iteration:02d}.patch").write_text(diff, encoding="utf-8")
    return bool(diff)


def read_dependency_files(workspace: Path) -> dict[str, str]:
    return {
        name: (workspace / name).read_text(encoding="utf-8") if (workspace / name).exists() else ""
        for name in ("pyproject.toml",)
    }


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


def fallback_observation(
    iteration: int,
    report: dict[str, object],
    score: float,
    accepted: bool,
    solver_changed: bool,
    hypothesis: str,
    error: str | None,
) -> str:
    metrics = []
    for item in compact_report(report).get("results", []):
        info = item.get("final_info", {}) if isinstance(item, dict) else {}
        if not isinstance(info, dict):
            continue
        metrics.append(
            f"- Problem `{item.get('problem_id')}`: pairwise_CA_RMSD_mean="
            f"`{info.get('pairwise_ca_rmsd_mean')}`, Rg_mean=`{info.get('radius_of_gyration_mean')}`, "
            f"conformers=`{info.get('num_conformers_generated')}`, finite=`{info.get('coordinate_finite')}`"
        )
    evidence = "\n".join(metrics) if metrics else "- No per-problem metrics were available."
    status = "supported" if accepted else "not supported"
    changed = "changed" if solver_changed else "did not change"
    return (
        "## Evidence\n\n"
        f"- Iteration `{iteration}` {changed} `solver.py` and produced score proxy `{score}`.\n"
        f"- Runner accepted: `{accepted}`. Error: `{error}`.\n"
        f"{evidence}\n\n"
        "## Supported/Rejected\n\n"
        f"The hypothesis was `{status}` by the bounded internal proxy. "
        f"Hypothesis excerpt: {hypothesis.strip()[:500] or 'No hypothesis text recorded.'}\n\n"
        "## Risks\n\n"
        "- The proxy is not the official hidden score and can still overvalue simple geometric diversity.\n"
        "- Sequence-only geometry remains a coarse approximation without learned priors or force-field relaxation.\n\n"
        "## Open Questions\n\n"
        "- Which observed metric changes correlate with the official hidden score is unknown.\n"
        "- Whether optional public dependencies would materially improve the bounded solver is untested in this run."
    )


def reflect_on_iteration(
    workspace: Path,
    iteration: int,
    model: str,
    base_url: str | None,
    report: dict[str, object] | None,
    score: float,
    accepted: bool,
    solver_changed: bool,
    hypothesis: str,
    error: str | None,
) -> tuple[str, object | None]:
    if report is None:
        observation = (
            f"Iteration {iteration} produced no complete suite report. "
            f"accepted={accepted}, score_proxy={score}, solver_changed={solver_changed}, error={error}"
        )
        (workspace / f"observation_{iteration:02d}.md").write_text(observation, encoding="utf-8")
        return observation, None

    prompt = (
        f"Reflect on iteration {iteration} of the AI4S protein ensemble agent.\n\n"
        "Literature context is available in literature_review.md and was used during design.\n\n"
        f"Hypothesis:\n{hypothesis[:3000]}\n\n"
        f"Runner decision: accepted={accepted}, score_proxy={score}, solver_changed={solver_changed}, error={error}\n\n"
        f"Experiment report JSON:\n{json.dumps(compact_report(report), indent=2)}\n\n"
        "Write a concise observation with these sections: Evidence, Supported/Rejected, Risks, Open Questions. "
        "Record facts and uncertainty only; do not prescribe the next strategy and do not include hidden chain-of-thought."
    )
    result = build_reflection_agent(workspace, model, base_url).run_sync(prompt)
    observation = result.final_answer or ""
    if not observation.strip() or stopped_for_max_tokens(getattr(result, "events_path", None)):
        observation = fallback_observation(iteration, report, score, accepted, solver_changed, hypothesis, error)
    (workspace / f"observation_{iteration:02d}.md").write_text(observation, encoding="utf-8")
    return observation, result


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
    iteration_root = workspace / "iteration_runs"
    iteration_root.mkdir(parents=True, exist_ok=True)
    baseline_report = run_suite(workspace / "solver.py", iteration_root / "baseline", solver_rounds, clean=True)
    best_score = proxy_score(baseline_report)
    (workspace / "baseline_result.json").write_text(
        json.dumps(compact_report(baseline_report), indent=2),
        encoding="utf-8",
    )

    for iteration in range(1, iterations + 1):
        shutil.copy2(best_solver, workspace / "solver.py")
        solver_before = (workspace / "solver.py").read_text(encoding="utf-8")
        write_iteration_context(workspace, iteration, iterations, best_score, history)
        result = None
        report: dict[str, object] | None = None
        report_path: Path | None = None
        error: str | None = None
        solver_changed = False
        dependency_changed = False
        warnings: list[str] = []
        agent = build_agent(workspace, model, base_url)
        try:
            dependency_before = read_dependency_files(workspace)
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
            dependency_changed = write_dependency_diff(workspace, iteration, dependency_before, read_dependency_files(workspace))
            hit_max_tokens = stopped_for_max_tokens(getattr(result, "events_path", None))
            if hit_max_tokens:
                warnings.append("agent response hit max_tokens after required artifacts were present; experiment was still evaluated")
            research_plan = load_research_plan(workspace)
            if not research_plan.strip():
                raise RuntimeError("agent did not write required research_plan.md")
            hypothesis = load_hypothesis(workspace)
            if not hypothesis.strip():
                raise RuntimeError("agent did not write required hypothesis.md")
            report_path = iteration_root / f"iteration_{iteration:02d}"
            report = run_suite(workspace / "solver.py", report_path, solver_rounds, clean=True)
            score = proxy_score(report)
            violations = hard_gate_violations(report)
            accepted = bool(report.get("ok")) and not violations and score >= best_score
            observation, reflection = reflect_on_iteration(
                workspace=workspace,
                iteration=iteration,
                model=model,
                base_url=base_url,
                report=report,
                score=score,
                accepted=accepted,
                solver_changed=solver_changed,
                hypothesis=hypothesis,
                error=None,
            )
            if accepted:
                best_score = score
                shutil.copy2(workspace / "solver.py", best_solver)
            else:
                shutil.copy2(best_solver, workspace / "solver.py")
        except Exception as exc:
            score = -1.0
            accepted = False
            error = str(exc)
            research_plan = load_research_plan(workspace)
            hypothesis = load_hypothesis(workspace)
            observation = (
                f"Iteration {iteration} failed before a complete accepted experiment. "
                f"solver_changed={solver_changed}, dependency_changed={dependency_changed}. Error: {error}"
            )
            reflection = None
            (workspace / f"observation_{iteration:02d}.md").write_text(observation, encoding="utf-8")
            shutil.copy2(best_solver, workspace / "solver.py")

        item = IterationResult(
            iteration=iteration,
            accepted=accepted,
            score=score,
            solver_changed=solver_changed,
            dependency_changed=dependency_changed,
            report_path=str(report_path) if report_path is not None else None,
            run_id=getattr(result, "run_id", None),
            stop_reason=getattr(result, "stop_reason", None),
            metrics=getattr(result, "metrics", None),
            events_path=getattr(result, "events_path", None),
            final_answer=(getattr(result, "final_answer", "") or "")[:4000],
            artifact_validation=getattr(result, "artifact_validation", None),
            research_plan=research_plan,
            hypothesis=hypothesis,
            observation=observation,
            reflection_run_id=getattr(reflection, "run_id", None),
            reflection_stop_reason=getattr(reflection, "stop_reason", None),
            reflection_metrics=getattr(reflection, "metrics", None),
            reflection_events_path=getattr(reflection, "events_path", None),
            warnings=warnings,
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
            "research_planning",
            iteration=item.iteration,
            research_plan=item.research_plan,
        )
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
            dependency_changed=item.dependency_changed,
            run_id=item.run_id,
            stop_reason=item.stop_reason,
            metrics=item.metrics,
            events_path=item.events_path,
            final_answer=item.final_answer,
            artifact_validation=item.artifact_validation,
            warnings=item.warnings,
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
            observation=item.observation,
            reflection_run_id=item.reflection_run_id,
            reflection_stop_reason=item.reflection_stop_reason,
            reflection_metrics=item.reflection_metrics,
            reflection_events_path=item.reflection_events_path,
            warnings=item.warnings,
            error=item.error,
        )


def append_literature_audit(agent_log: Path, literature: dict[str, object]) -> None:
    append_log(
        agent_log,
        "literature_search",
        summary="OpenAlex literature retrieval completed before agent iteration.",
        source=literature.get("source"),
        queries=literature.get("queries"),
        paper_count=literature.get("paper_count"),
        papers=[
            {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "query": paper.get("query"),
            }
            for paper in literature.get("papers", [])
            if isinstance(paper, dict)
        ][:12],
        errors=literature.get("errors"),
    )


def append_environment_audit(agent_log: Path, environment: dict[str, object]) -> None:
    append_log(
        agent_log,
        "environment_probe",
        summary="Environment probe completed before agent iteration.",
        cpu_count=environment.get("cpu_count"),
        disk=environment.get("disk"),
        python_modules=environment.get("python_modules"),
        commands={
            key: value.get("available")
            for key, value in (environment.get("commands") or {}).items()
            if isinstance(value, dict)
        },
    )


def append_memory_audit(agent_log: Path, memory_summary: dict[str, object] | None) -> None:
    if memory_summary is None:
        return
    append_log(
        agent_log,
        "memory_update",
        summary="Long-term project memory updated after this run.",
        memory_summary=memory_summary,
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
    write_memory_context(ROOT, workspace)
    environment = write_environment_report(workspace, ROOT)
    literature = collect_literature(workspace)

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
    memory_summary = update_memory(ROOT, workspace, history, report, literature, environment)
    append_environment_audit(out_root / "submission" / "agent.log", environment)
    append_literature_audit(out_root / "submission" / "agent.log", literature)
    if history:
        append_iteration_audit(out_root / "submission" / "agent.log", history)
        report["agent_iterations"] = [item.__dict__ for item in history]
    append_memory_audit(out_root / "submission" / "agent.log", memory_summary)
    validation = validate_submission(out_root / "submission")
    package_output(out_root / "submission", out_root / "output.zip")
    report["validation"] = validation
    report["ok"] = validation["ok"]
    report["literature"] = {
        "source": literature.get("source"),
        "paper_count": literature.get("paper_count"),
        "queries": literature.get("queries"),
        "errors": literature.get("errors"),
    }
    report["environment"] = {
        "cpu_count": environment.get("cpu_count"),
        "disk": environment.get("disk"),
        "python_modules": environment.get("python_modules"),
        "commands": {
            key: value.get("available")
            for key, value in (environment.get("commands") or {}).items()
            if isinstance(value, dict)
        },
    }
    report["memory_summary"] = memory_summary
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
    archive_dir = archive_output(out_root)
    report["archive_dir"] = str(archive_dir)
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
    shutil.copy2(out_root / "run_report.json", archive_dir / "run_report.json")
    shutil.copy2(out_root / "technical_report.md", archive_dir / "technical_report.md")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
