import math
from pathlib import Path

import numpy as np
import pytest

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def write_ca_cif(path: Path, problem_id: str, conf_idx: int, sequence: str, ca_coords: np.ndarray) -> None:
    lines = [
        f"data_{problem_id}_conf{conf_idx}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    for i, aa in enumerate(sequence, start=1):
        if i - 1 >= len(ca_coords):
            break
        x, y, z = ca_coords[i - 1]
        res_name = AA3.get(aa, "UNK")
        lines.append(
            f"ATOM {i} C CA {res_name} A {i} {x:.3f} {y:.3f} {z:.3f} 1.00 20.00 A {i} 1"
        )
    lines.append("#")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_uniform_ca(seq_len: int, spacing: float = 3.8, phase: float = 0.0) -> np.ndarray:
    coords = np.zeros((seq_len, 3), dtype=np.float64)
    angle = phase
    for i in range(seq_len):
        coords[i] = [spacing * i + math.cos(angle) * 0.1, math.sin(angle) * 0.1, 0.0]
        angle += 1.0
    fixed = np.zeros_like(coords)
    fixed[0] = coords[0]
    for i in range(1, seq_len):
        d = coords[i] - fixed[i - 1]
        n = np.linalg.norm(d) or 1.0
        fixed[i] = fixed[i - 1] + d * spacing / n
    return fixed


@pytest.fixture
def make_cif():
    return write_ca_cif


@pytest.fixture
def uniform_ca():
    return make_uniform_ca
