from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .memory import append_jsonl, memory_dir
from .run_suite import ROOT


def parse_score_json(text: str | None, path: str | None) -> dict[str, Any]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--score-json must decode to a JSON object")
    return value


def record_score(
    root: Path,
    score: float | None,
    score1: float | None,
    score2: float | None,
    success: bool | None,
    error_msg: str,
    submission: str,
    notes: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    mem = memory_dir(root)
    score_json = raw.get("scoreJson") if isinstance(raw.get("scoreJson"), dict) else {}
    item = {
        "timestamp_unix": int(time.time()),
        "score": score if score is not None else raw.get("score"),
        "score1": score1 if score1 is not None else score_json.get("score1"),
        "score2": score2 if score2 is not None else score_json.get("score2"),
        "success": success if success is not None else raw.get("success"),
        "errorMsg": error_msg or raw.get("errorMsg") or "",
        "submission": submission,
        "notes": notes,
        "raw": raw,
    }
    append_jsonl(mem / "scores.jsonl", item)
    with (mem / "observations.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join([
                f"## Official Score {item['timestamp_unix']}",
                "",
                f"- Submission: `{submission}`",
                f"- Score: `{item.get('score')}`, score1: `{item.get('score1')}`, score2: `{item.get('score2')}`",
                f"- Success: `{item.get('success')}`, error: `{item.get('errorMsg')}`",
                f"- Notes: {notes or 'none'}",
                "",
            ])
        )
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=float, default=None)
    parser.add_argument("--score1", type=float, default=None)
    parser.add_argument("--score2", type=float, default=None)
    parser.add_argument("--success", choices=("true", "false"), default=None)
    parser.add_argument("--error-msg", default="")
    parser.add_argument("--submission", default="outputs/latest/output.zip")
    parser.add_argument("--notes", default="")
    parser.add_argument("--score-json", default=None, help="Raw official JSON result.")
    parser.add_argument("--score-json-file", default=None, help="Path to a file containing the raw official JSON result.")
    args = parser.parse_args()
    success = None if args.success is None else args.success == "true"
    item = record_score(
        ROOT,
        args.score,
        args.score1,
        args.score2,
        success,
        args.error_msg,
        args.submission,
        args.notes,
        parse_score_json(args.score_json, args.score_json_file),
    )
    print(json.dumps(item, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
