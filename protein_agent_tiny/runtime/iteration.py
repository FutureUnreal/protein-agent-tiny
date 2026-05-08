from __future__ import annotations
import concurrent.futures
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .. import problems as problems_mod
from ..run_suite import DEFAULT_OUTPUT, ROOT, package_output
from ..memory import update_memory, write_memory_context
from ..literature import collect_literature
from ..environment import write_environment_report
from ..validate import validate_submission
from ..archive import archive_output
from ..report import build_report
from ..scoring.proxy import score_submission, hard_gate
from ..scoring.cif import parse_submission_dir
from .contracts import RuntimeConfig, BootstrapResult, IterationResult
from .workspace import prepare_workspace
from .audit import append_event
from .agents import build_bootstrap_agent, build_improve_agent, build_reflect_agent
from .solver_env import resolve_solver_python


# --- helpers ---

GOAL_BOOTSTRAP = """Iteration 0 (bootstrap) for the AI4S protein ensemble competition.

This workspace has no existing solver_pkg/. You are creating the FIRST VERSION.

Required deliverables (all four MUST exist before this iteration is considered complete):
- research_plan.md (≥200 bytes): mode, facts considered, candidate architectures, chosen action, validation plan
- hypothesis.md (≥80 bytes, ≤12 bullet lines)
- notes.md
- solver_pkg/cli.py (≥200 bytes) — must accept the CLI: --problem-id ID --sequence SEQ --num-conformers N --optimization-rounds R --out-dir DIR
- solver_pkg/pipeline.py (≥200 bytes) — implementation
- solver_pkg/.pipeline_ready — sentinel file written ONLY after cli.py works locally; must contain the literal token `ready` (one byte minimum, e.g. `ready\n`)

After writing the package, run a smoke test:
  python print_sequence.py 1 > /tmp/seq.txt   (or read inline)
  python solver_pkg/cli.py --problem-id 1 --sequence <SEQ> --num-conformers 2 --optimization-rounds {solver_rounds} --out-dir smoke

When the smoke test produces valid CIF files, write `ready\n` (or any non-empty content) to `solver_pkg/.pipeline_ready`. The sentinel must NOT be empty — the artifact contract requires at least one byte.
"""

GOAL_IMPROVE = """Iteration {iteration} of {total_iterations} for the AI4S protein ensemble competition.

A working solver_pkg/ exists. Read it and the iteration_context.json, then make a MINIMAL bounded change.
Write research_plan.md and hypothesis.md as required by SKILL.md before editing solver_pkg/.

CLI contract: python solver_pkg/cli.py --problem-id 1 --sequence SEQ --num-conformers 2 --optimization-rounds {solver_rounds} --out-dir smoke
"""


def _progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _run_with_heartbeat(label: str, fn, interval_seconds: int = 30):
    _progress(f"{label} started")
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        while True:
            try:
                result = future.result(timeout=interval_seconds)
                elapsed = int(time.time() - started)
                _progress(f"{label} finished in {elapsed}s")
                return result
            except concurrent.futures.TimeoutError:
                elapsed = int(time.time() - started)
                _progress(f"{label} still running ({elapsed}s elapsed)")


def _expected_problems_map(workspace_or_root: Path) -> dict:
    """Returns {problem_id: sequence_length} for problems 1, 2, 3."""
    out = {}
    pdir = workspace_or_root / "problems" if (workspace_or_root / "problems").exists() else (ROOT / "data" / "problems")
    for spec in problems_mod.load_problems(pdir):
        out[spec.problem_id] = len(spec.sequence)
    return out


def _solver_cli_path(workspace: Path) -> Path:
    candidate = workspace / "solver_pkg" / "cli.py"
    if candidate.exists():
        return candidate
    return workspace / "solver.py"


def _write_workspace_solver_shim(workspace: Path) -> None:
    shim = workspace / "solver.py"
    if shim.exists():
        return
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n"
        "import runpy, sys\n"
        "from pathlib import Path\n"
        "cli = Path(__file__).parent / 'solver_pkg' / 'cli.py'\n"
        "if not cli.exists():\n"
        "    raise SystemExit('solver_pkg/cli.py not present; bootstrap-agent did not produce a pipeline.')\n"
        "sys.argv[0] = str(cli)\n"
        "runpy.run_path(str(cli), run_name='__main__')\n",
        encoding="utf-8",
    )


def _smoke_test_cli(workspace: Path, sequence: str) -> bool:
    _progress("bootstrap smoke test started")
    cli = workspace / "solver_pkg" / "cli.py"
    if not cli.exists():
        _progress("bootstrap smoke test skipped: solver_pkg/cli.py missing")
        return False
    smoke_dir = workspace / "_bootstrap_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir, ignore_errors=True)
    solver_py = resolve_solver_python(workspace).python
    cmd = [
        solver_py, str(cli),
        "--problem-id", "1",
        "--sequence", sequence,
        "--num-conformers", "2",
        "--optimization-rounds", "1",
        "--out-dir", str(smoke_dir),
    ]
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(workspace)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,
            cwd=str(workspace),
            env=env,
            text=True,
        )
        (workspace / "_bootstrap_smoke_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (workspace / "_bootstrap_smoke_stderr.log").write_text(result.stderr or "", encoding="utf-8")
        ok = result.returncode == 0 and any(smoke_dir.glob("*_pred.cif"))
        detail = "" if ok else f" rc={result.returncode}; see _bootstrap_smoke_stderr.log"
        _progress(f"bootstrap smoke test {'passed' if ok else 'failed'}{detail}")
        return ok
    except Exception as exc:
        _progress(f"bootstrap smoke test failed: {exc}")
        return False


def _write_iteration_context(
    workspace: Path,
    iteration: int,
    total: int,
    best_score: float,
    history: list,
    current_proxy=None,
) -> None:
    """Snapshot the state the improve-agent must read before deciding.

    `current_proxy` is the latest ProxyReport: either the baseline (before
    iteration 1) or the report from the previously-finished iteration. Without
    it the agent cannot diagnose bottlenecks, only see the aggregate best.
    """
    if current_proxy is not None:
        diagnostics = {
            "current_score_proxy": current_proxy.score,
            "current_hard_gate_violations": list(current_proxy.hard_gate_violations),
            "current_per_problem": current_proxy.per_problem,
            "current_mode": current_proxy.mode,
        }
    else:
        diagnostics = {
            "current_score_proxy": None,
            "current_hard_gate_violations": [],
            "current_per_problem": {},
            "current_mode": None,
        }
    ctx = {
        "iteration": iteration,
        "total_iterations": total,
        "best_score_proxy": best_score,
        **diagnostics,
        "accepted_history": [
            {
                "iteration": h.iteration,
                "score_proxy": h.score,
                "accepted": h.accepted,
                "research_plan": (h.research_plan or "")[:2000],
                "hypothesis": (h.hypothesis or "")[:2000],
                "observation": (h.observation or "")[:2000],
                "error": h.error,
            }
            for h in history
        ],
        "literature_review_path": "literature_review.md",
        "memory_context_path": "memory_context.md",
        "environment_report_path": "environment_report.md",
        "score_proxy_definition": "Internal selection proxy from real CIF coordinates: diversity/rg/spacing/clash/dihedral/pca/precision/finite. NOT the official hidden score.",
    }
    (workspace / "iteration_context.json").write_text(json.dumps(ctx, indent=2), encoding="utf-8")


# --- phases ---

def bootstrap_phase(cfg: RuntimeConfig, workspace: Path, problems_dir: Path, sample_sequence: str) -> BootstrapResult:
    """Run bootstrap-agent up to cfg.bootstrap_max_attempts.

    On failure: record per-attempt errors and return success=False. We do NOT
    fabricate CIFs to disguise an unproductive run. The audit trail (agent.log
    + bootstrap_attempt_*.md) is the truthful artifact. The final submission
    may legitimately contain zero CIFs; the competition rules accept per-problem
    zero scores without invalidating the run.
    """
    sentinel = workspace / "solver_pkg" / ".pipeline_ready"
    if sentinel.exists():
        _progress("existing solver_pkg sentinel found; validating current pipeline")
        # Fast path: re-validate the existing pipeline before declaring success.
        if _smoke_test_cli(workspace, sample_sequence):
            _write_workspace_solver_shim(workspace)
            _progress("existing solver_pkg accepted")
            return BootstrapResult(success=True, attempts=0, sentinel_written=True, emergency_invoked=False, error=None)
        # Sentinel exists but smoke fails -> treat as broken; remove sentinel and re-bootstrap.
        try:
            sentinel.unlink()
        except OSError:
            pass

    attempt_errors: list[str] = []
    last_error = None
    for attempt in range(1, cfg.bootstrap_max_attempts + 1):
        _progress(f"bootstrap-agent attempt {attempt}/{cfg.bootstrap_max_attempts}")
        try:
            agent = build_bootstrap_agent(cfg, workspace, ROOT / "runs")
            result = _run_with_heartbeat(
                f"bootstrap-agent attempt {attempt}",
                lambda: agent.run_sync(GOAL_BOOTSTRAP.format(solver_rounds=cfg.solver_rounds)),
            )
            (workspace / f"bootstrap_attempt_{attempt:02d}.md").write_text(
                result.final_answer or "", encoding="utf-8"
            )
            if sentinel.exists() and _smoke_test_cli(workspace, sample_sequence):
                _write_workspace_solver_shim(workspace)
                _progress(f"bootstrap-agent attempt {attempt} accepted")
                return BootstrapResult(success=True, attempts=attempt, sentinel_written=True, emergency_invoked=False, error=None)
            last_error = "sentinel or smoke test failed"
        except Exception as exc:
            last_error = str(exc)
            _progress(f"bootstrap-agent attempt {attempt} failed: {last_error[:300]}")
        attempt_errors.append(f"attempt_{attempt}: {last_error}")

    # Honest failure: no CIF fabrication. Caller will see emergency_invoked=False
    # and finalize with whatever CIFs (if any) exist in workspace, plus a complete agent.log.
    return BootstrapResult(
        success=False, attempts=cfg.bootstrap_max_attempts,
        sentinel_written=False, emergency_invoked=False,
        error="; ".join(attempt_errors) or last_error,
    )


def _solver_diff(workspace: Path, iteration: int, before_files: dict, after_files: dict) -> bool:
    parts = []
    for name in sorted(set(before_files) | set(after_files)):
        if before_files.get(name, "") == after_files.get(name, ""):
            continue
        parts.append("".join(difflib.unified_diff(
            before_files.get(name, "").splitlines(keepends=True),
            after_files.get(name, "").splitlines(keepends=True),
            fromfile=f"{name}.before.iter{iteration:02d}",
            tofile=f"{name}.after.iter{iteration:02d}",
        )))
    if any(parts):
        (workspace / f"solver_diff_{iteration:02d}.patch").write_text("".join(parts), encoding="utf-8")
        return True
    return False


def _read_solver_pkg(workspace: Path) -> dict:
    pkg = workspace / "solver_pkg"
    out = {}
    if pkg.exists():
        for p in sorted(pkg.rglob("*.py")):
            out[str(p.relative_to(workspace))] = p.read_text(encoding="utf-8")
    return out


def _proxy_for_workspace(workspace: Path, problems_dir: Path) -> tuple:
    """Run solver_pkg via run_suite, then score real CIFs. Returns (ProxyReport, raw_run_report, iteration_root)."""
    from ..run_suite import run_suite as run_suite_fn
    iteration_root = workspace / "iteration_runs" / f"score_{int(time.time())}"
    raw = _run_with_heartbeat(
        f"proxy run_suite {iteration_root.name}",
        lambda: run_suite_fn(workspace / "solver.py", iteration_root, 1, clean=True),
    )
    submission = iteration_root / "submission"
    expected = _expected_problems_map(workspace)
    proxy = score_submission(submission, expected)
    _progress(
        f"proxy score={proxy.score:.4f} mode={proxy.mode} "
        f"violations={len(proxy.hard_gate_violations)}"
    )
    return proxy, raw, iteration_root


def _restore_solver_pkg_from_best(workspace: Path, best_pipeline: Path) -> None:
    """Defensive rollback: restore solver_pkg from best_pipeline regardless of solver_pkg's current state."""
    if not best_pipeline.exists():
        return
    target = workspace / "solver_pkg"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(best_pipeline, target)


def improve_phase(
    cfg: RuntimeConfig, workspace: Path, problems_dir: Path
) -> list:
    _progress("improve phase started")
    history = []
    best_pipeline = workspace / "best_pipeline"
    if best_pipeline.exists():
        shutil.rmtree(best_pipeline, ignore_errors=True)
    shutil.copytree(workspace / "solver_pkg", best_pipeline)

    # If the full-sequence baseline run fails (for example solver_pkg passes a
    # short-sequence smoke test but crashes on the longer competition sequences),
    # skip improve iterations and let final packaging emit whatever CIFs survived.
    try:
        _progress("baseline full-sequence proxy evaluation")
        baseline_proxy, _, _ = _proxy_for_workspace(workspace, problems_dir)
    except Exception as exc:
        _progress(f"baseline full-sequence proxy evaluation failed: {str(exc)[:300]}")
        (workspace / "baseline_result.json").write_text(
            json.dumps({"score": -1.0, "error": f"baseline_run_failed: {exc}"}, indent=2),
            encoding="utf-8",
        )
        return history
    best_score = baseline_proxy.score
    (workspace / "baseline_result.json").write_text(
        json.dumps({"score": best_score, "violations": list(baseline_proxy.hard_gate_violations), "per_problem": baseline_proxy.per_problem}, indent=2),
        encoding="utf-8",
    )

    current_proxy = baseline_proxy
    for iteration in range(1, cfg.iterations + 1):
        _progress(f"iteration {iteration}/{cfg.iterations} started; best_score={best_score:.4f}")
        before_files = _read_solver_pkg(workspace)
        _write_iteration_context(
            workspace, iteration, cfg.iterations, best_score, history,
            current_proxy=current_proxy,
        )
        try:
            agent = build_improve_agent(cfg, workspace, ROOT / "runs")
            result = _run_with_heartbeat(
                f"improve-agent iteration {iteration}/{cfg.iterations}",
                lambda: agent.run_sync(GOAL_IMPROVE.format(
                    iteration=iteration, total_iterations=cfg.iterations, solver_rounds=cfg.solver_rounds,
                )),
            )
            (workspace / f"agent_final_answer_{iteration:02d}.md").write_text(result.final_answer or "", encoding="utf-8")
            after_files = _read_solver_pkg(workspace)
            solver_changed = _solver_diff(workspace, iteration, before_files, after_files)
            _progress(f"iteration {iteration}: solver_changed={solver_changed}")

            research_plan = (workspace / "research_plan.md").read_text(encoding="utf-8") if (workspace / "research_plan.md").exists() else ""
            hypothesis = (workspace / "hypothesis.md").read_text(encoding="utf-8") if (workspace / "hypothesis.md").exists() else ""
            if not research_plan.strip() or not hypothesis.strip():
                raise RuntimeError("agent did not write research_plan.md / hypothesis.md")

            proxy, _, score_dir = _proxy_for_workspace(workspace, problems_dir)
            score = proxy.score
            accepted = (not proxy.hard_gate_violations) and score >= best_score
            current_proxy = proxy

            obs_text = _reflect(cfg, workspace, iteration, proxy, accepted, hypothesis)
            if accepted:
                best_score = score
                if best_pipeline.exists():
                    shutil.rmtree(best_pipeline, ignore_errors=True)
                shutil.copytree(workspace / "solver_pkg", best_pipeline)
            else:
                _restore_solver_pkg_from_best(workspace, best_pipeline)
            _progress(
                f"iteration {iteration}/{cfg.iterations} "
                f"{'accepted' if accepted else 'rejected'}; score={score:.4f}; "
                f"best_score={best_score:.4f}"
            )

            history.append(IterationResult(
                iteration=iteration,
                accepted=accepted,
                score=score,
                solver_changed=solver_changed,
                dependency_changed=False,
                report_path=str(score_dir),
                run_id=getattr(result, "run_id", None),
                stop_reason=getattr(result, "stop_reason", None),
                metrics=getattr(result, "metrics", None),
                events_path=getattr(result, "events_path", None),
                final_answer=(getattr(result, "final_answer", "") or "")[:4000],
                artifact_validation=getattr(result, "artifact_validation", None),
                research_plan=research_plan,
                hypothesis=hypothesis,
                observation=obs_text,
                warnings=(),
                error=None,
            ))
        except Exception as exc:
            _restore_solver_pkg_from_best(workspace, best_pipeline)
            _progress(f"iteration {iteration}/{cfg.iterations} failed: {str(exc)[:300]}")
            history.append(IterationResult(
                iteration=iteration,
                accepted=False,
                score=-1.0,
                solver_changed=False,
                dependency_changed=False,
                report_path=None,
                run_id=None,
                stop_reason=None,
                metrics=None,
                events_path=None,
                final_answer="",
                artifact_validation=None,
                research_plan="",
                hypothesis="",
                observation=f"Iteration {iteration} failed: {exc}",
                warnings=(),
                error=str(exc),
            ))
    return history


def _reflect(cfg: RuntimeConfig, workspace: Path, iteration: int, proxy, accepted: bool, hypothesis: str) -> str:
    try:
        agent = build_reflect_agent(cfg, workspace, ROOT / "runs")
        prompt = (
            f"Reflect on iteration {iteration}. accepted={accepted}, score={proxy.score}, "
            f"violations={proxy.hard_gate_violations}, per_problem={proxy.per_problem}, "
            f"hypothesis={hypothesis[:1500]}.\n"
            "Sections: Evidence, Supported/Rejected, Risks, Open Questions."
        )
        result = _run_with_heartbeat(
            f"reflect-agent iteration {iteration}",
            lambda: agent.run_sync(prompt),
        )
        text = (result.final_answer or "").strip()
        if text:
            (workspace / f"observation_{iteration:02d}.md").write_text(text, encoding="utf-8")
            return text
    except Exception:
        pass
    fallback = (
        f"Iteration {iteration} fallback observation. accepted={accepted}, score={proxy.score}, "
        f"violations={list(proxy.hard_gate_violations)}."
    )
    (workspace / f"observation_{iteration:02d}.md").write_text(fallback, encoding="utf-8")
    return fallback


# --- top-level entry ---


def _write_required_audit_skeleton(agent_log: Path, mode: str, reason: str) -> None:
    """Write the five required agent.log event types as audit-only placeholders
    when no real iteration produced them. Used by package-only mode and the
    bootstrap-failed branch so agent.log stays evaluator-readable while still
    honestly stating that no science was executed.
    """
    common = {"audit_only": True, "mode": mode, "reason": reason}
    append_event(agent_log, "literature_search", **common, paper_count=0, queries=[])
    append_event(agent_log, "approach_decision", **common,
                 summary=f"{mode}: no agent science executed; this run is for audit/packaging only.")
    append_event(agent_log, "code_evolution", **common,
                 summary=f"{mode}: no code evolution occurred.")
    append_event(agent_log, "experiment_run", **common,
                 summary=f"{mode}: no experiments executed.")
    append_event(agent_log, "experiment_observation", **common,
                 summary=f"{mode}: no observations recorded.")


def run(cfg: RuntimeConfig) -> int:
    workspace = cfg.workspace
    out_root = cfg.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    submission = out_root / "submission"
    _progress(
        "agent runner started: "
        f"iterations={cfg.iterations}, solver_rounds={cfg.solver_rounds}, "
        f"max_minutes={cfg.max_minutes}, workspace={workspace}, out={out_root}, "
        f"model={cfg.model}"
    )

    if cfg.skip_agent:
        _progress("skip-agent package-only mode started")
        # Package-only audit mode. Produces no CIFs because fabricating placeholders
        # would misrepresent agent capability. Useful for CI / packaging smoke tests;
        # not a valid scientific submission.
        if submission.exists():
            shutil.rmtree(submission)
        submission.mkdir(parents=True)
        agent_log = submission / "agent.log"
        _write_required_audit_skeleton(
            agent_log, mode="package_only",
            reason="--skip-agent flag set; LLM and bootstrap were intentionally bypassed.",
        )
        append_event(agent_log, "skip_agent_invoked", reason="--skip-agent flag")
        validation = validate_submission(submission)
        package_output(submission, out_root / "output.zip")
        report = {
            "ok": validation["ok"],
            "validation": validation,
            "skip_agent": True,
            "package_only": True,
            "output_zip": str(out_root / "output.zip"),
            "cif_count": 0,
            "results": [],
        }
        (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
        archive_output(out_root)
        _progress(f"skip-agent package-only mode finished; output_zip={out_root / 'output.zip'}")
        return 0 if validation["ok"] else 1

    # Full agent path
    _progress("preparing workspace")
    prepare_workspace(workspace, ROOT)
    _progress("writing memory context")
    write_memory_context(ROOT, workspace)
    environment = _run_with_heartbeat(
        "environment probe",
        lambda: write_environment_report(workspace, ROOT),
    )
    literature = _run_with_heartbeat(
        "literature collection",
        lambda: collect_literature(workspace),
    )

    problems_dir = workspace / "problems"
    sample_sequence = next(iter(problems_mod.load_problems(problems_dir))).sequence

    _progress("bootstrap phase started")
    boot_result = bootstrap_phase(cfg, workspace, problems_dir, sample_sequence)
    _progress(
        f"bootstrap phase finished: success={boot_result.success}, "
        f"attempts={boot_result.attempts}, error={boot_result.error}"
    )

    history: list = []
    final_run_error: str | None = None
    if boot_result.success:
        history = improve_phase(cfg, workspace, problems_dir)
        _progress(f"improve phase finished: iterations_recorded={len(history)}")

    # Final packaging — no fabrication path. Four honest outcomes:
    #   1. skip_agent (handled above): zero CIFs + audit_only agent.log.
    #   2. Bootstrap failed: zero CIFs; agent.log records every attempt's error.
    #   3. Bootstrap succeeded, run_suite finished: submission has CIFs for each
    #      problem whose solver finished successfully.
    #   4. Bootstrap succeeded, run_suite crashed mid-pass: whatever CIFs were
    #      copied before the crash stay; the crashed problem simply has fewer or
    #      zero CIFs; final_run_error is recorded in agent.log + final_run_error.txt.
    if submission.exists():
        shutil.rmtree(submission)
    submission.mkdir(parents=True)

    if boot_result.success:
        try:
            from ..run_suite import run_suite as run_suite_fn
            _run_with_heartbeat(
                "final run_suite packaging pass",
                lambda: run_suite_fn(workspace / "solver.py", out_root, cfg.solver_rounds, clean=True),
            )
        except Exception as exc:
            final_run_error = str(exc)
            _progress(f"final run_suite packaging pass failed: {final_run_error[:300]}")
            (workspace / "final_run_error.txt").write_text(final_run_error, encoding="utf-8")
            # run_suite copies CIFs into submission/ as it goes; whatever survived
            # is honest output. Do NOT overwrite or fabricate.

    agent_log = submission / "agent.log"
    append_event(agent_log, "literature_search", source=literature.get("source"), paper_count=literature.get("paper_count"), queries=literature.get("queries"))
    append_event(agent_log, "environment_probe", cpu_count=environment.get("cpu_count"), python_modules=environment.get("python_modules"))
    append_event(
        agent_log, "approach_decision",
        bootstrap_success=boot_result.success,
        bootstrap_attempts=boot_result.attempts,
        bootstrap_error=boot_result.error,
        final_run_error=final_run_error,
    )
    if not boot_result.success:
        # Bootstrap failed entirely: still write the required event skeleton so
        # the audit log is parseable, but tag the mode so consumers can tell
        # this run produced no real experiments.
        for required_evt in ("code_evolution", "experiment_run", "experiment_observation"):
            append_event(
                agent_log, required_evt,
                audit_only=True, mode="bootstrap_failed",
                summary=f"bootstrap failed after {boot_result.attempts} attempts; no {required_evt} occurred.",
                bootstrap_error=boot_result.error,
            )
    for h in history:
        # Embed research_plan and hypothesis excerpts so the audit log itself
        # carries research-depth evidence, not just aggregate metrics.
        append_event(
            agent_log, "code_evolution",
            iteration=h.iteration,
            accepted=h.accepted,
            score_proxy=h.score,
            solver_changed=h.solver_changed,
            research_plan_excerpt=(h.research_plan or "")[:800],
            hypothesis_excerpt=(h.hypothesis or "")[:500],
            error=h.error,
        )
        append_event(agent_log, "experiment_run", iteration=h.iteration, report_path=h.report_path, score_proxy=h.score)
        append_event(agent_log, "experiment_observation", iteration=h.iteration, observation=h.observation[:4000], error=h.error)

    # Backfill required event types if the path above did not produce them
    # AND the bootstrap_failed skeleton was not already written.
    # Without this, a successful bootstrap whose final run_suite crashed
    # before any improve iteration would produce an agent.log missing
    # code_evolution / experiment_run / experiment_observation, and
    # validate_submission would reject the submission.
    if boot_result.success and not history:
        for evt in ("code_evolution", "experiment_run", "experiment_observation"):
            append_event(
                agent_log, evt,
                audit_only=True,
                mode="final_run_failed" if final_run_error else "no_iterations_executed",
                summary=(
                    f"No {evt} occurred. "
                    + (f"final_run_error={final_run_error[:500]}" if final_run_error else "improve_phase produced zero iterations.")
                ),
            )

    memory_summary = update_memory(ROOT, workspace, history, {"results": [], "ok": True}, literature, environment)
    append_event(agent_log, "memory_update", memory_summary=memory_summary)

    cif_count = len(list(submission.glob("*_pred.cif")))
    validation = validate_submission(submission)
    package_output(submission, out_root / "output.zip")
    report = {
        "ok": validation["ok"],
        "validation": validation,
        "bootstrap": asdict(boot_result),
        "history_count": len(history),
        "best_score": max((h.score for h in history if h.accepted), default=-1.0),
        "output_zip": str(out_root / "output.zip"),
        "cif_count": cif_count,
        "final_run_error": final_run_error,
        "results": [],
    }
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "technical_report.md").write_text(build_report(out_root), encoding="utf-8")
    archive_output(out_root)
    _progress(
        f"agent runner finished: ok={validation['ok']}, cif_count={cif_count}, "
        f"output_zip={out_root / 'output.zip'}"
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 1
