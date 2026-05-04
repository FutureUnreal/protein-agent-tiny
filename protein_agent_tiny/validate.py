from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


REQUIRED_LOG_EVENTS = {"literature_search", "approach_decision", "code_evolution", "experiment_run", "experiment_observation"}
REQUIRED_PROBLEM_IDS = {"1", "2", "3"}
MAX_CONFORMERS_PER_PROBLEM = 10


def check_cif(path: Path) -> tuple[bool, list[str]]:
    """Deep validation: surface text checks + structured CIF parsing.

    A CIF can pass naive regex tests (presence of `_atom_site`, an `ATOM` line,
    the substring ` CA `) yet still be unparseable in practice — for example
    when `label_comp_id` uses single-letter amino acid codes that real mmCIF
    parsers reject. We delegate the structural check to scoring.cif.parse_cif
    so this validator agrees with the proxy scorer.
    """
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    # Cheap surface checks first; these catch grossly malformed files.
    if "_atom_site" not in text:
        errors.append("missing _atom_site")
    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM ")]
    if not atom_lines:
        errors.append("no ATOM records")
    if re.search(r"\b(nan|inf|-inf)\b", text, re.IGNORECASE):
        errors.append("contains NaN/Inf")

    # Structural parse — must agree with scoring.cif.
    from .scoring.cif import parse_cif as _structural_parse
    parsed = _structural_parse(path)
    if parsed.errors:
        for err in parsed.errors:
            errors.append(f"parse:{err}")
    if parsed.ca_coords.shape[0] == 0:
        # parse_cif already reports "no_ca_atoms" in errors; surface a clear
        # message even if errors was somehow empty.
        if "parse:no_ca_atoms" not in errors:
            errors.append("structural parse found 0 CA atoms")

    return not errors, errors


def check_agent_log(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, ["agent.log missing"]
    errors: list[str] = []
    events: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_no}: invalid JSON")
            continue
        event_type = str(event.get("event_type", ""))
        if event_type:
            events.add(event_type)
    missing = REQUIRED_LOG_EVENTS - events
    if missing:
        errors.append(f"missing log event types: {sorted(missing)}")
    return not errors, errors


def validate_submission(submission_dir: Path) -> dict[str, object]:
    """Validate the official submission contract.

    Hard-fail (ok=False):
      - agent.log missing / malformed / missing required event types (赛题: missing log disqualifies)
      - per-CIF format errors (filename pattern, _atom_site, NaN/Inf, no ATOM, no CA)
      - any conformer index > MAX_CONFORMERS_PER_PROBLEM

    Warning (does NOT fail ok, but recorded in result):
      - Missing valid conformers for one or more problems (FAQ Q4: per-problem
        zero score is allowed; we don't fabricate to hide it).
      - Submission containing zero CIFs but valid agent.log (honest failure run).
    """
    result: dict[str, object] = {"ok": True, "files": [], "agent_log": {}, "problem_coverage": {}, "warnings": []}
    cif_files = sorted(submission_dir.glob("*_conf*_pred.cif"))
    valid_per_problem: dict[str, int] = {}
    all_per_problem: dict[str, int] = {}

    if not cif_files:
        result["warnings"].append("no CIF files in submission (honest failure record; agent.log still required)")
    for path in cif_files:
        match = re.fullmatch(r"([123])_conf([1-9]\d*)_pred\.cif", path.name)
        ok, errors = check_cif(path)
        if not match:
            ok = False
            errors.append("filename must be {1|2|3}_confN_pred.cif")
        else:
            pid = match.group(1)
            all_per_problem[pid] = all_per_problem.get(pid, 0) + 1
            if int(match.group(2)) > MAX_CONFORMERS_PER_PROBLEM:
                ok = False
                errors.append(f"conformer index exceeds {MAX_CONFORMERS_PER_PROBLEM}")
            if ok:
                valid_per_problem[pid] = valid_per_problem.get(pid, 0) + 1
        result["files"].append({"file": path.name, "ok": ok, "errors": errors})
        result["ok"] = bool(result["ok"]) and ok

    # Hard-fail: any problem exceeded the conformer cap.
    over_count = {pid: n for pid, n in all_per_problem.items() if n > MAX_CONFORMERS_PER_PROBLEM}
    if over_count:
        result["ok"] = False
        result["error"] = f"problems exceed {MAX_CONFORMERS_PER_PROBLEM} conformers: {over_count}"

    # Soft warning: missing problems (FAQ Q4 explicitly allows zero score per problem).
    missing_problems = sorted(REQUIRED_PROBLEM_IDS - set(valid_per_problem.keys()))
    if missing_problems:
        result["warnings"].append(f"missing valid conformers for problems {missing_problems} (per-problem score will be 0; this is a legal honest-failure outcome)")
    result["problem_coverage"] = {
        "required": sorted(REQUIRED_PROBLEM_IDS),
        "valid_per_problem": valid_per_problem,
        "all_per_problem": all_per_problem,
        "missing": missing_problems,
    }

    # Hard requirement (competition rule): agent.log must be present and well-formed.
    log_ok, log_errors = check_agent_log(submission_dir / "agent.log")
    result["agent_log"] = {"ok": log_ok, "errors": log_errors}
    result["ok"] = bool(result["ok"]) and log_ok
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate_submission(Path(args.submission_dir))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("PASS" if result["ok"] else "FAIL")
        for item in result.get("files", []):
            print(f"{'OK' if item['ok'] else 'FAIL'} {item['file']} {'; '.join(item['errors'])}")
        print(f"agent.log: {'OK' if result['agent_log']['ok'] else 'FAIL'}")
        for error in result["agent_log"]["errors"]:
            print(f"  - {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
