#!/usr/bin/env python3
"""Default bounded solver for the AI4S protein ensemble task.

The agent is expected to improve this file. Keep the CLI contract stable:
`python solver.py --problem-id 1 --sequence ... --num-conformers 4 --out-dir run`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

HYDROPHOBIC = set("AILMFWVY")
FLEXIBLE = set("GPSND")
CHARGED = set("DEKRH")


def log_event(path: Path, event_type: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"event_type": event_type, "timestamp_unix": int(time.time()), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def residue_bias(aa: str) -> float:
    bias = 0.0
    if aa in HYDROPHOBIC:
        bias -= 0.020
    if aa in FLEXIBLE:
        bias += 0.040
    if aa in CHARGED:
        bias += 0.012
    return bias


def ca_trace(sequence: str, conformer_index: int, round_index: int) -> list[tuple[float, float, float]]:
    """Generate a compact-ish finite CA trace with near-constant CA spacing."""
    coords: list[tuple[float, float, float]] = []
    n = len(sequence)
    phase = 0.59 * conformer_index + 0.17 * round_index
    radius = 8.5 + 0.35 * conformer_index
    axial_scale = 0.42 + 0.03 * ((round_index + conformer_index) % 4)
    angle = phase

    for i, aa in enumerate(sequence):
        angle += 1.63 + residue_bias(aa) + 0.025 * math.sin(i / 29.0 + phase)
        fold = math.sin(i / max(24.0, math.sqrt(n)) + phase)
        x = radius * math.cos(angle) + 7.0 * math.sin(i / 41.0 + phase)
        y = radius * math.sin(angle) + 5.5 * math.cos(i / 37.0 + phase)
        z = axial_scale * i + 18.0 * fold
        coords.append((x, y, z))

    return normalize_ca_spacing(coords, 3.8)


def normalize_ca_spacing(
    coords: list[tuple[float, float, float]], target: float
) -> list[tuple[float, float, float]]:
    if len(coords) < 2:
        return coords
    fixed = [coords[0]]
    for point in coords[1:]:
        prev = fixed[-1]
        dx, dy, dz = point[0] - prev[0], point[1] - prev[1], point[2] - prev[2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        fixed.append((prev[0] + dx * target / norm, prev[1] + dy * target / norm, prev[2] + dz * target / norm))
    return fixed


def radius_of_gyration(coords: list[tuple[float, float, float]]) -> float:
    n = len(coords)
    cx = sum(p[0] for p in coords) / n
    cy = sum(p[1] for p in coords) / n
    cz = sum(p[2] for p in coords) / n
    return math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 for x, y, z in coords) / n)


def ca_distance_stats(coords: list[tuple[float, float, float]]) -> tuple[float, float]:
    distances = []
    for a, b in zip(coords, coords[1:]):
        dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
        distances.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return (min(distances), max(distances)) if distances else (0.0, 0.0)


def centered_rmsd(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> float:
    n = min(len(a), len(b))
    ca = [sum(p[k] for p in a[:n]) / n for k in range(3)]
    cb = [sum(p[k] for p in b[:n]) / n for k in range(3)]
    total = 0.0
    for pa, pb in zip(a[:n], b[:n]):
        total += sum(((pa[k] - ca[k]) - (pb[k] - cb[k])) ** 2 for k in range(3))
    return math.sqrt(total / n)


def ensemble_metrics(traces: list[list[tuple[float, float, float]]]) -> dict[str, object]:
    pairwise = [
        centered_rmsd(traces[i], traces[j])
        for i in range(len(traces))
        for j in range(i + 1, len(traces))
    ]
    rgs = [radius_of_gyration(trace) for trace in traces]
    min_ca, max_ca = ca_distance_stats(traces[0]) if traces else (0.0, 0.0)
    return {
        "pairwise_ca_rmsd_values": [round(v, 3) for v in pairwise],
        "pairwise_ca_rmsd_mean": round(sum(pairwise) / len(pairwise), 3) if pairwise else 0.0,
        "pairwise_ca_rmsd_std": round((sum((v - sum(pairwise) / len(pairwise)) ** 2 for v in pairwise) / len(pairwise)) ** 0.5, 3) if pairwise else 0.0,
        "radius_of_gyration_mean": round(sum(rgs) / len(rgs), 3) if rgs else 0.0,
        "radius_of_gyration_values": [round(v, 3) for v in rgs],
        "min_ca_distance": round(min_ca, 3),
        "max_ca_distance": round(max_ca, 3),
    }


def backbone_atoms(ca_coords: list[tuple[float, float, float]]) -> list[list[tuple[str, str, tuple[float, float, float]]]]:
    residues = []
    n = len(ca_coords)
    for i, ca in enumerate(ca_coords):
        prev_ca = ca_coords[max(0, i - 1)]
        next_ca = ca_coords[min(n - 1, i + 1)]
        tx, ty, tz = next_ca[0] - prev_ca[0], next_ca[1] - prev_ca[1], next_ca[2] - prev_ca[2]
        norm = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
        tx, ty, tz = tx / norm, ty / norm, tz / norm
        ux, uy, uz = -ty, tx, 0.0
        un = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / un, uy / un, uz / un
        residues.append([
            ("N", "N", (ca[0] - 1.45 * tx + 0.18 * ux, ca[1] - 1.45 * ty + 0.18 * uy, ca[2] - 1.45 * tz)),
            ("CA", "C", ca),
            ("C", "C", (ca[0] + 1.52 * tx - 0.12 * ux, ca[1] + 1.52 * ty - 0.12 * uy, ca[2] + 1.52 * tz)),
            ("O", "O", (ca[0] + 1.52 * tx + 0.83 * ux, ca[1] + 1.52 * ty + 0.83 * uy, ca[2] + 1.52 * tz + 0.18)),
        ])
    return residues


def write_mmcif(path: Path, problem_id: str, conf_idx: int, sequence: str, ca_coords: list[tuple[float, float, float]]) -> None:
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
    atom_id = 1
    for res_idx, (aa, atoms) in enumerate(zip(sequence, backbone_atoms(ca_coords)), start=1):
        res_name = AA3.get(aa, "UNK")
        for atom_name, element, (x, y, z) in atoms:
            lines.append(
                f"ATOM {atom_id} {element} {atom_name} {res_name} A {res_idx} "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 20.00 A {res_idx} 1"
            )
            atom_id += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(problem_id: str, sequence: str, num_conformers: int, out_dir: Path, optimization_rounds: int) -> dict[str, object]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_traces = [ca_trace(sequence, i, 1) for i in range(1, num_conformers + 1)]
    candidate_count = len(best_traces)

    for round_idx in range(2, optimization_rounds + 1):
        candidate = [ca_trace(sequence, i, round_idx) for i in range(1, num_conformers + 1)]
        candidate_count += len(candidate)
        if ensemble_metrics(candidate)["pairwise_ca_rmsd_mean"] >= ensemble_metrics(best_traces)["pairwise_ca_rmsd_mean"]:
            best_traces = candidate

    for i, trace in enumerate(best_traces, start=1):
        write_mmcif(out_dir / f"{problem_id}_conf{i}_pred.cif", problem_id, i, sequence, trace)

    info = {
        "problem_id": problem_id,
        "sequence_length": len(sequence),
        "conformer_count": num_conformers,
        "num_conformers_generated": len(best_traces),
        "optimization_rounds_requested": optimization_rounds,
        "optimization_rounds_completed": optimization_rounds,
        "candidate_count": candidate_count,
        "coordinate_finite": all(math.isfinite(v) for trace in best_traces for point in trace for v in point),
        "approach_used": "bounded_sequence_only_backbone_baseline",
        "data_governance": "Used provided sequence only; no competition MD/crystal/NMR inputs.",
        "runtime_seconds": round(time.time() - started, 3),
        **ensemble_metrics(best_traces),
    }
    (out_dir / "final_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-conformers", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--optimization-rounds", type=int, default=1)
    args = parser.parse_args()
    info = run(args.problem_id, args.sequence, min(args.num_conformers, 10), Path(args.out_dir), max(1, args.optimization_rounds))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
