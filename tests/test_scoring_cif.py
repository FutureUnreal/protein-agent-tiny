from pathlib import Path

import numpy as np
import pytest

from protein_agent_tiny.scoring.cif import ParsedCif, parse_cif, parse_submission_dir

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_parse_minimal_ca_only():
    parsed = parse_cif(FIXTURES / "minimal_ca_only.cif")
    # Bad filename (no problem_id_confN_pred pattern) so expect bad_filename error
    assert "bad_filename" in parsed.errors
    assert parsed.residue_count == 5
    assert parsed.mode == "ca_only"
    assert parsed.ca_coords.shape == (5, 3)
    assert np.all(np.isfinite(parsed.ca_coords))


def test_parse_full_backbone():
    parsed = parse_cif(FIXTURES / "full_backbone.cif")
    assert parsed.residue_count == 5
    assert parsed.mode == "full_backbone"
    assert parsed.backbone_coords is not None
    for atom in ("N", "CA", "C", "O"):
        assert atom in parsed.backbone_coords
        assert parsed.backbone_coords[atom].shape == (5, 3)


def test_parse_bad_filename():
    parsed = parse_cif(FIXTURES / "minimal_ca_only.cif")
    assert "bad_filename" in parsed.errors


def test_parse_submission_dir_duplicate_ensemble():
    dup_dir = FIXTURES / "duplicate_ensemble"
    result = parse_submission_dir(dup_dir)
    assert "1" in result
    conformers = result["1"]
    assert len(conformers) == 2
    assert conformers[0].conf_idx < conformers[1].conf_idx
    for c in conformers:
        assert c.problem_id == "1"
        assert c.residue_count == 5
