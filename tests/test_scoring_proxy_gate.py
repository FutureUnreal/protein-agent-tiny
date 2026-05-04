import math
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_uniform_ca, write_ca_cif
from protein_agent_tiny.scoring.proxy import ProxyReport, hard_gate, score_submission
from protein_agent_tiny.scoring.cif import parse_submission_dir


SEQ5 = "MKTAY"
SEQ8 = "MKTAYGSR"


def _build_ok_submission(tmp_path: Path, problem_id: str, sequence: str, n_conf: int) -> Path:
    sub = tmp_path / problem_id
    sub.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_conf + 1):
        ca = make_uniform_ca(len(sequence), spacing=3.8, phase=float(i) * 0.3)
        write_ca_cif(sub / f"{problem_id}_conf{i}_pred.cif", problem_id, i, sequence, ca)
    return sub


# 5 conformers with good geometry → score > 0, no violations
def test_ok_conformers_no_violations(tmp_path):
    sub = _build_ok_submission(tmp_path, "1", SEQ5, 5)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert isinstance(report, ProxyReport)
    assert report.score > 0
    assert len(report.hard_gate_violations) == 0


# all conformers identical → all_conformers_duplicate violation, score = 0
def test_duplicate_conformers(tmp_path):
    sub = tmp_path / "dup"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=3.8)
    for i in range(1, 4):
        write_ca_cif(sub / f"1_conf{i}_pred.cif", "1", i, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert any("all_conformers_duplicate" in v for v in report.hard_gate_violations)
    assert report.score == 0.0


# conformer with NaN coordinates → nonfinite error, hard gate violation
def test_nonfinite_coords(tmp_path):
    sub = tmp_path / "nan"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=3.8)
    ca[2, 0] = float("nan")
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert any("nonfinite" in v for v in report.hard_gate_violations)


# severe clash (CA at <3.0 Å non-adjacent) → severe_clash violation
def test_severe_clash(tmp_path):
    sub = tmp_path / "clash"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ8), spacing=3.8)
    # Move last residue to within 2.0 Å of residue 2 (non-adjacent, |7-1|=6)
    ca[7] = ca[1] + np.array([0.5, 0.0, 0.0])
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ8, ca)
    report = score_submission(sub, {"1": len(SEQ8)})
    assert any("severe_clash" in v for v in report.hard_gate_violations)


# spacing 5.0 Å (deviates >1.0 from 3.8) → spacing_mean violation
def test_spacing_violation(tmp_path):
    sub = tmp_path / "spacing"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=5.0)
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert any("spacing_mean" in v for v in report.hard_gate_violations)


# Additional: hard_gate function accepts parsed_by_problem dict directly
def test_hard_gate_empty_conformers():
    violations = hard_gate({"1": []}, {"1": 5})
    assert any("no_conformers" in v for v in violations)
