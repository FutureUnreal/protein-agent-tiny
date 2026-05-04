from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBLEM_DIR = ROOT / "data" / "problems"


@dataclass(frozen=True)
class Problem:
    problem_id: str
    source_name: str
    sequence: str
    conformer_count: int


def load_problem(path: Path) -> Problem:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"{path} must contain exactly one official problem record")
    data = records[0]
    sequences = data["sequences"]
    if not isinstance(sequences, list) or len(sequences) != 1:
        raise ValueError(f"{path} must contain exactly one single-chain sequence entry")
    protein_chain = sequences[0]["proteinChain"]
    chain_count = int(protein_chain.get("count", 1))
    if chain_count != 1:
        raise ValueError(f"{path} declares proteinChain.count={chain_count}; this tiny baseline expects a single chain")
    return Problem(
        problem_id=path.stem,
        source_name=str(data["name"]),
        sequence=protein_chain["sequence"].strip().upper(),
        conformer_count=int(data["conformer_count"]),
    )


def load_problems(problem_dir: Path = PROBLEM_DIR) -> list[Problem]:
    return [load_problem(path) for path in sorted(problem_dir.glob("*.json"))]
