from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from .problems import Problem, load_problems
from .validate import validate_submission


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "latest"


@lru_cache(maxsize=1)
def competition_sequences() -> tuple[tuple[str, str], ...]:
    return tuple((problem.problem_id, problem.sequence) for problem in load_problems())


def sanitize_log_text(value: str) -> str:
    text = value
    project_root = ROOT.resolve()
    path_variants = {
        str(project_root),
        project_root.as_posix(),
        str(project_root).replace("\\", "/"),
    }
    for variant in sorted(path_variants, key=len, reverse=True):
        if variant:
            text = text.replace(variant, "<PROJECT_ROOT>")
    for problem_id, sequence in competition_sequences():
        if len(sequence) >= 20:
            text = text.replace(sequence, f"<PROBLEM_{problem_id}_SEQUENCE_LEN_{len(sequence)}>")
    return text


def sanitize_for_log(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_log_text(value)
    if isinstance(value, Path):
        return sanitize_log_text(str(value))
    if isinstance(value, dict):
        return {key: sanitize_for_log(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_log(item) for item in value]
    return value


def append_log(path: Path, event_type: str, **payload: object) -> None:
    event = sanitize_for_log({"event_type": event_type, "timestamp_unix": int(time.time()), **payload})
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def solver_command(solver: Path, problem: Problem, out_dir: Path, rounds: int) -> list[str]:
    return [
        sys.executable,
        str(solver),
        "--problem-id",
        problem.problem_id,
        "--sequence",
        problem.sequence,
        "--num-conformers",
        str(min(problem.conformer_count, 10)),
        "--optimization-rounds",
        str(max(1, rounds)),
        "--out-dir",
        str(out_dir),
    ]


def solver_command_summary(solver: Path, problem: Problem, out_dir: Path, rounds: int) -> dict[str, object]:
    return {
        "python": Path(sys.executable).name,
        "solver": str(solver),
        "problem_id": problem.problem_id,
        "sequence_length": len(problem.sequence),
        "num_conformers": min(problem.conformer_count, 10),
        "optimization_rounds": max(1, rounds),
        "out_dir": str(out_dir),
    }


def package_output(submission_dir: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.iterdir()):
            if path.name == "agent.log" or path.name.endswith("_pred.cif"):
                archive.write(path, arcname=path.name)


def run_suite(solver: Path, out_root: Path, rounds: int, clean: bool = False) -> dict[str, object]:
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    submission_dir = out_root / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    agent_log = submission_dir / "agent.log"
    if agent_log.exists():
        agent_log.unlink()

    append_log(
        agent_log,
        "literature_search",
        summary="Competition-tiny run records allowed data policy. No competition MD/crystal/NMR inputs were used.",
        allowed_public_data=["RCSB PDB", "AlphaFold DB", "UniProt/UniRef/MGnify", "unrelated public MD benchmark datasets"],
        forbidden_data="This competition's original MD trajectories, crystal structures, or NMR ensembles.",
    )
    append_log(
        agent_log,
        "approach_decision",
        summary="Run the current solver.py sequence-conditioned ensemble generator, then validate and package official output.zip.",
        solver=str(solver),
    )
    append_log(
        agent_log,
        "code_evolution",
        summary="Using the current solver.py implementation for this run. Agent-improved runs should point to the workspace solver.",
        solver=str(solver),
    )

    results: list[dict[str, object]] = []
    for problem in load_problems():
        run_dir = out_root / "runs" / problem.problem_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = solver_command(solver, problem, run_dir, rounds)
        started = time.time()
        proc = subprocess.run(command, text=True, capture_output=True, timeout=3600)
        elapsed = round(time.time() - started, 3)
        (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        info_path = run_dir / "final_info.json"
        final_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        append_log(
            agent_log,
            "experiment_run",
            problem_id=problem.problem_id,
            command_summary=solver_command_summary(solver, problem, run_dir, rounds),
            returncode=proc.returncode,
            elapsed_seconds=elapsed,
            final_info=final_info,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Solver failed for problem {problem.problem_id}: {proc.stderr[-2000:]}")
        for cif in sorted(run_dir.glob(f"{problem.problem_id}_conf*_pred.cif"))[:10]:
            shutil.copy2(cif, submission_dir / cif.name)
        results.append({"problem_id": problem.problem_id, "elapsed_seconds": elapsed, "final_info": final_info})

    validation = validate_submission(submission_dir)
    file_validation = {
        "ok": bool(validation.get("files")) and all(bool(item.get("ok")) for item in validation.get("files", []) if isinstance(item, dict)),
        "files": validation.get("files", []),
        "agent_log": "validated after appending experiment_observation",
    }
    append_log(
        agent_log,
        "experiment_observation",
        summary="Validated generated CIF files after running all problems.",
        validation_ok=file_validation["ok"],
        validation=file_validation,
    )
    validation = validate_submission(submission_dir)
    output_zip = out_root / "output.zip"
    package_output(submission_dir, output_zip)
    model_metadata = {
        "artifact_type": "model_metadata",
        "solver": str(solver),
        "training_status": "inference_only_or_bounded_agent_solver",
        "notes": (
            "This tiny project treats solver.py as the model artifact. "
            "If an agent implements training, it should write weights under outputs/latest/model/."
        ),
    }
    model_dir = out_root / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    report = {
        "ok": validation["ok"],
        "solver": str(solver),
        "model_metadata": str(model_dir / "metadata.json"),
        "out_root": str(out_root),
        "output_zip": str(output_zip),
        "results": results,
        "validation": validation,
    }
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default=str(ROOT / "protein_agent_tiny" / "solver.py"))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = run_suite(Path(args.solver), Path(args.out), args.rounds, args.clean)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
