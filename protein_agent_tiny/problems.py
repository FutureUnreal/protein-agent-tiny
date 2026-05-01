from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBLEM_DIR = ROOT / "data" / "problems"


@dataclass(frozen=True)
class Problem:
    problem_id: str
    sequence: str
    conformer_count: int


def load_problem(path: Path) -> Problem:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Problem(
        problem_id=path.stem,
        sequence=data["proteinChain"]["sequence"].strip().upper(),
        conformer_count=int(data["conformer_count"]),
    )


def load_problems(problem_dir: Path = PROBLEM_DIR) -> list[Problem]:
    return [load_problem(path) for path in sorted(problem_dir.glob("*.json"))]
