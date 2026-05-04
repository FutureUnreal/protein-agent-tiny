from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Filename pattern: {problem_id}_conf{N}_pred.cif
_FILENAME_RE = re.compile(r"^(.+)_conf(\d+)_pred\.cif$")

# Column order produced by current solver's _atom_site loop:
# group_PDB id type_symbol label_atom_id label_comp_id label_asym_id
# label_seq_id Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv
# auth_asym_id auth_seq_id pdbx_PDB_model_num
_KNOWN_COLS = [
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "auth_asym_id",
    "auth_seq_id",
    "pdbx_PDB_model_num",
]

_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class ParsedCif:
    problem_id: str
    conf_idx: int
    residue_count: int
    ca_coords: np.ndarray          # (N, 3) float64
    backbone_coords: Optional[dict]  # keys among {"N","CA","C","O"}; None when ca_only
    mode: str                      # "ca_only" | "full_backbone"
    errors: tuple


def _strip_prefix(col_name: str) -> str:
    """Remove _atom_site. prefix if present."""
    return col_name.split(".")[-1] if "." in col_name else col_name


def _parse_atom_site_loop(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Parse a single _atom_site loop_ block.

    Returns (records, parse_errors).
    """
    errors: list[str] = []
    # Collect column headers
    col_names: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("_atom_site"):
            col_names.append(_strip_prefix(stripped))
            i += 1
        else:
            break

    if not col_names:
        errors.append("no_atom_site_columns")
        return [], errors

    idx = {name: i for i, name in enumerate(col_names)}

    records: list[dict] = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#") or line.startswith("loop_") or line.startswith("_"):
            break
        # Skip comment lines inside data block
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < len(col_names):
            continue
        rec: dict = {}
        for col, cidx in idx.items():
            rec[col] = parts[cidx]
        records.append(rec)

    return records, errors


def parse_cif(path: Path) -> "ParsedCif":
    """Parse a mmCIF file produced by the solver.

    Never raises — errors are recorded in ParsedCif.errors.
    """
    errors: list[str] = []

    # --- filename parsing ---
    m = _FILENAME_RE.match(path.name)
    if m:
        problem_id = m.group(1)
        conf_idx = int(m.group(2))
    else:
        problem_id = ""
        conf_idx = 0
        errors.append("bad_filename")

    # --- file reading ---
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        errors.append("unreadable")
        return ParsedCif(
            problem_id=problem_id,
            conf_idx=conf_idx,
            residue_count=0,
            ca_coords=np.empty((0, 3), dtype=np.float64),
            backbone_coords=None,
            mode="ca_only",
            errors=tuple(errors),
        )

    lines = text.splitlines()

    # --- locate _atom_site loop ---
    atom_site_start: Optional[int] = None
    for idx_line, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "loop_":
            # Check if next non-empty line is an _atom_site field
            for j in range(idx_line + 1, min(idx_line + 5, len(lines))):
                nxt = lines[j].strip()
                if nxt.startswith("_atom_site"):
                    atom_site_start = idx_line + 1
                    break
                if nxt and not nxt.startswith("#"):
                    break
            if atom_site_start is not None:
                break

    if atom_site_start is None:
        errors.append("no_atom_site_loop")
        return ParsedCif(
            problem_id=problem_id,
            conf_idx=conf_idx,
            residue_count=0,
            ca_coords=np.empty((0, 3), dtype=np.float64),
            backbone_coords=None,
            mode="ca_only",
            errors=tuple(errors),
        )

    records, parse_errs = _parse_atom_site_loop(lines[atom_site_start:])
    errors.extend(parse_errs)

    # --- collect coordinates by residue ---
    # res_key -> {atom_name -> (x, y, z)}
    res_map: dict[str, dict[str, tuple[float, float, float]]] = {}
    res_order: list[str] = []

    for rec in records:
        group = rec.get("group_PDB", "ATOM")
        if group not in ("ATOM", "HETATM"):
            continue
        atom_name = rec.get("label_atom_id", "")
        seq_id = rec.get("label_seq_id", "")
        if not seq_id or not atom_name:
            continue
        try:
            x = float(rec.get("Cartn_x", "nan"))
            y = float(rec.get("Cartn_y", "nan"))
            z = float(rec.get("Cartn_z", "nan"))
        except ValueError:
            x = y = z = float("nan")

        if seq_id not in res_map:
            res_map[seq_id] = {}
            res_order.append(seq_id)
        # Keep first occurrence of each atom type per residue
        if atom_name not in res_map[seq_id]:
            res_map[seq_id][atom_name] = (x, y, z)

    # --- build CA coords in residue order ---
    ca_list: list[tuple[str, tuple[float, float, float]]] = []
    for seq_id in res_order:
        atoms = res_map[seq_id]
        if "CA" in atoms:
            ca_list.append((seq_id, atoms["CA"]))

    if not ca_list:
        errors.append("no_ca_atoms")
        return ParsedCif(
            problem_id=problem_id,
            conf_idx=conf_idx,
            residue_count=0,
            ca_coords=np.empty((0, 3), dtype=np.float64),
            backbone_coords=None,
            mode="ca_only",
            errors=tuple(errors),
        )

    ca_coords = np.array([list(xyz) for _, xyz in ca_list], dtype=np.float64)

    # Check for non-finite coordinates
    if not np.all(np.isfinite(ca_coords)):
        errors.append("nonfinite")

    residue_count = len(ca_list)
    ca_seq_ids = [sid for sid, _ in ca_list]

    # --- check full backbone ---
    # full_backbone requires N, CA, C, O for every residue that has a CA
    has_full_backbone = all(
        all(a in res_map[sid] for a in _BACKBONE_ATOMS)
        for sid in ca_seq_ids
    )

    if has_full_backbone:
        mode = "full_backbone"
        backbone_coords: Optional[dict] = {
            atom: np.array(
                [list(res_map[sid][atom]) for sid in ca_seq_ids],
                dtype=np.float64,
            )
            for atom in _BACKBONE_ATOMS
        }
    else:
        mode = "ca_only"
        backbone_coords = None

    return ParsedCif(
        problem_id=problem_id,
        conf_idx=conf_idx,
        residue_count=residue_count,
        ca_coords=ca_coords,
        backbone_coords=backbone_coords,
        mode=mode,
        errors=tuple(errors),
    )


def parse_submission_dir(submission_dir: Path) -> dict[str, list["ParsedCif"]]:
    """Scan *_conf*_pred.cif files and return dict keyed by problem_id."""
    result: dict[str, list[ParsedCif]] = {}

    cif_files = sorted(submission_dir.glob("*_conf*_pred.cif"))
    for cif_path in cif_files:
        parsed = parse_cif(cif_path)
        pid = parsed.problem_id if parsed.problem_id else cif_path.stem
        result.setdefault(pid, [])
        result[pid].append(parsed)

    # Sort each list by conf_idx
    for pid in result:
        result[pid].sort(key=lambda p: p.conf_idx)

    return result
