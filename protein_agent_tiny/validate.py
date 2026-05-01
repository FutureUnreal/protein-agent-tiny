from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


REQUIRED_LOG_EVENTS = {"literature_search", "approach_decision", "code_evolution", "experiment_run", "experiment_observation"}


def check_cif(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "_atom_site" not in text:
        errors.append("missing _atom_site")
    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM ")]
    if not atom_lines:
        errors.append("no ATOM records")
    if not any(" CA " in f" {line} " for line in atom_lines):
        errors.append("no CA atoms")
    if re.search(r"\b(nan|inf|-inf)\b", text, re.IGNORECASE):
        errors.append("contains NaN/Inf")
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
    result: dict[str, object] = {"ok": True, "files": [], "agent_log": {}}
    cif_files = sorted(submission_dir.glob("*_conf*_pred.cif"))
    if not cif_files:
        result["ok"] = False
        result["error"] = "no CIF files"
        return result
    seen: dict[str, int] = {}
    for path in cif_files:
        match = re.fullmatch(r"([123])_conf([1-9]\d*)_pred\.cif", path.name)
        ok, errors = check_cif(path)
        if not match:
            ok = False
            errors.append("filename must be {1|2|3}_confN_pred.cif")
        else:
            pid = match.group(1)
            seen[pid] = seen.get(pid, 0) + 1
            if int(match.group(2)) > 10:
                ok = False
                errors.append("conformer index exceeds 10")
        result["files"].append({"file": path.name, "ok": ok, "errors": errors})
        result["ok"] = bool(result["ok"]) and ok
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
