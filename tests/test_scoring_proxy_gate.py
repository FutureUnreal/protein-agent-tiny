from pathlib import Path

import numpy as np

from tests.conftest import make_uniform_ca, write_ca_cif
from protein_agent_tiny.scoring.proxy import ProxyReport, hard_gate, score_submission


SEQ5 = "MKTAY"
SEQ8 = "MKTAYGSR"


def _build_ok_submission(tmp_path: Path, problem_id: str, sequence: str, n_conf: int) -> Path:
    sub = tmp_path / problem_id
    sub.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_conf + 1):
        ca = make_uniform_ca(len(sequence), spacing=3.8, phase=float(i) * 0.3)
        write_ca_cif(sub / f"{problem_id}_conf{i}_pred.cif", problem_id, i, sequence, ca)
    return sub


def test_ok_conformers_no_violations(tmp_path):
    sub = _build_ok_submission(tmp_path, "1", SEQ5, 5)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert isinstance(report, ProxyReport)
    assert report.score > 0
    assert len(report.hard_gate_violations) == 0


def test_duplicate_conformers(tmp_path):
    sub = tmp_path / "dup"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=3.8)
    for i in range(1, 4):
        write_ca_cif(sub / f"1_conf{i}_pred.cif", "1", i, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert any("all_conformers_duplicate" in v for v in report.hard_gate_violations)
    assert any("all_conformers_duplicate" in v for v in report.geometry_violations)
    assert report.score == 0.0


def test_nonfinite_coords(tmp_path):
    sub = tmp_path / "nan"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=3.8)
    ca[2, 0] = float("nan")
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert any("nonfinite" in v for v in report.hard_gate_violations)
    assert any("nonfinite" in v for v in report.format_violations)


def test_severe_clash_hard_gate(tmp_path):
    sub = tmp_path / "clash"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ8), spacing=3.8)
    ca[7] = ca[1] + np.array([0.5, 0.0, 0.0])
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ8, ca)
    report = score_submission(sub, {"1": len(SEQ8)})
    assert any("severe_clash" in v for v in report.hard_gate_violations)
    assert any("severe_clash" in v for v in report.geometry_violations)
    assert report.score == 0.0


def test_soft_clash_penalty(tmp_path):
    sub = tmp_path / "soft_clash"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ8), spacing=3.8)
    ca[7] = ca[1] + np.array([0.0, 2.5, 0.0])
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ8, ca)
    report = score_submission(sub, {"1": len(SEQ8)})
    assert not report.hard_gate_violations
    assert any("soft_clash" in v for v in report.geometry_violations)
    assert report.per_problem["1"]["soft_clash_penalty"] == 0.3
    assert report.score > 0


def test_spacing_violation_is_soft(tmp_path):
    sub = tmp_path / "spacing"
    sub.mkdir()
    ca = make_uniform_ca(len(SEQ5), spacing=5.0)
    write_ca_cif(sub / "1_conf1_pred.cif", "1", 1, SEQ5, ca)
    report = score_submission(sub, {"1": len(SEQ5)})
    assert not any("spacing_mean" in v for v in report.hard_gate_violations)


def test_hard_gate_empty_conformers():
    violations = hard_gate({"1": []}, {"1": 5})
    assert any("no_conformers" in v for v in violations)
