from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cif import ParsedCif, parse_submission_dir
from .geometry import (
    ca_nonadjacent_min_distance,
    ca_spacing_stats,
    ensemble_pairwise_matrix,
    medoid_outlier_ratio,
    pca_effective_rank,
    pseudo_dihedral_smoothness,
    radius_of_gyration,
)


# Mirrors runtime.contracts.ProxyReport. The duplication keeps the scoring
# package free of any runtime/ import (one-way dependency: runtime -> scoring).
# Both classes are structurally identical so consumers can use duck typing.
# Keep field names and types in lockstep with runtime.contracts.ProxyReport
# and update tests/test_scoring_proxy_gate.py if the schema evolves.
@dataclass(frozen=True)
class ProxyReport:
    score: float
    per_problem: dict
    hard_gate_violations: tuple
    mode: str
    format_violations: tuple = ()
    geometry_violations: tuple = ()


# Proxy weight constants.
# Weights chosen to balance coverage-like (diversity), precision-like (outlier penalty),
# and structural plausibility (rg, spacing, clash, dihedral, pca, finite). Sum to 1.0.
W_DIVERSITY = 0.25
W_RG = 0.15
W_SPACING = 0.10
W_CLASH = 0.15
W_DIHEDRAL = 0.10
W_PCA = 0.10
W_PRECISION = 0.10
W_FINITE = 0.05


def _bounded_peak(value: float, low: float, target: float, high: float) -> float:
    if not math.isfinite(value) or value <= low:
        return 0.0
    if value <= target:
        return min(1.0, (value - low) / max(target - low, 1e-6))
    if value >= high:
        return 0.0
    return max(0.0, 1.0 - (value - target) / max(high - target, 1e-6))


def _rg_score(rg: float, n_residues: int) -> float:
    if rg <= 0 or n_residues <= 0:
        return 0.0
    upper = 2.8 * (n_residues ** 0.52)
    lower = 0.9 * (n_residues ** 0.35)
    if rg < lower * 0.5:
        return 0.0
    if rg <= upper:
        return 1.0 if rg >= lower else (rg - lower * 0.5) / max(lower - lower * 0.5, 1e-6)
    return max(0.0, 1.0 - (rg - upper) / max(1.2 * upper, 1e-6))


def _spacing_score(stats: dict) -> float:
    mean = stats.get("mean", 0.0)
    std = stats.get("std", 0.0)
    mean_error = abs(mean - 3.8)
    return max(0.0, 1.0 - (mean_error / 0.4) - (std / 0.5))


def _clash_score(min_nonadj: float) -> float:
    # >=4.0 A excellent, 3.0 A acceptable, 2.0-3.0 A is a soft penalty,
    # <2.0 A is a hard gate handled by hard_gate().
    if not math.isfinite(min_nonadj):
        return 1.0  # too few residues to evaluate
    if min_nonadj >= 4.0:
        return 1.0
    if min_nonadj >= 3.0:
        return (min_nonadj - 3.0) / 1.0
    if min_nonadj >= 2.0:
        return 0.3 * ((min_nonadj - 2.0) / 1.0)
    return 0.0


def _soft_clash_penalty(clash_vals: list[float]) -> float:
    finite = [v for v in clash_vals if math.isfinite(v)]
    if not finite:
        return 1.0
    soft_count = sum(1 for v in finite if 2.0 <= v < 3.0)
    if soft_count == 0:
        return 1.0
    ratio = soft_count / len(finite)
    return max(0.3, 1.0 - 0.7 * ratio)


def _conformer_uniqueness(parsed_list: list[ParsedCif]) -> tuple[float, float]:
    hashes = [np.round(p.ca_coords, 1).tobytes() for p in parsed_list if p.ca_coords.size]
    if len(hashes) <= 1:
        return 1.0, 1.0
    unique_fraction = len(set(hashes)) / len(hashes)
    return unique_fraction, max(0.3, unique_fraction)


def _is_format_violation(text: str) -> bool:
    markers = (
        "no_conformers",
        "bad_filename",
        "no_atom_site_columns",
        "no_ca_atoms",
        "nonfinite",
        "residue_count_mismatch",
        "no_ca",
    )
    return any(marker in text for marker in markers)


def _split_violations(violations: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    format_violations = tuple(v for v in violations if _is_format_violation(v))
    geometry_violations = tuple(v for v in violations if not _is_format_violation(v))
    return format_violations, geometry_violations


def _pca_score(pe: dict, n_conformers: int) -> float:
    rank = pe.get("rank", 1.0)
    if n_conformers <= 1:
        return 0.0
    max_useful = min(float(n_conformers), 6.0)
    return max(0.0, min(1.0, (rank - 1.0) / max(max_useful - 1.0, 1e-6)))


def hard_gate(parsed_by_problem: dict, expected_lengths: dict) -> tuple:
    """Return tuple of violation strings; empty = pass.

    expected_lengths: {problem_id: sequence_length}
    """
    violations = []
    for pid, parsed_list in parsed_by_problem.items():
        if not parsed_list:
            violations.append(f"{pid}:no_conformers")
            continue
        expected_len = expected_lengths.get(pid, 0)
        for parsed in parsed_list:
            if parsed.errors:
                for err in parsed.errors:
                    violations.append(f"{pid}:conf{parsed.conf_idx}:{err}")
            if expected_len > 0 and parsed.residue_count != expected_len:
                violations.append(
                    f"{pid}:conf{parsed.conf_idx}:residue_count_mismatch:"
                    f"{parsed.residue_count}!={expected_len}"
                )
            if parsed.ca_coords.size == 0:
                violations.append(f"{pid}:conf{parsed.conf_idx}:no_ca")
                continue
            min_nonadj = ca_nonadjacent_min_distance(parsed.ca_coords)
            if math.isfinite(min_nonadj) and min_nonadj < 2.0:
                violations.append(
                    f"{pid}:conf{parsed.conf_idx}:severe_clash_{min_nonadj:.2f}"
                )
    return tuple(violations)


def score_submission(submission_dir: Path, expected_problems: dict) -> ProxyReport:
    """Compute proxy score for a submission directory.

    expected_problems: {problem_id: sequence_length}
    """
    parsed_by_problem = parse_submission_dir(submission_dir)
    for pid in expected_problems:
        parsed_by_problem.setdefault(pid, [])

    violations = hard_gate(parsed_by_problem, expected_problems)
    soft_geometry_violations: list[str] = []

    modes: set = set()
    for parsed_list in parsed_by_problem.values():
        for p in parsed_list:
            # W2 fix: skip entries with fatal parse errors so mode aggregate
            # does not misreport a broken submission as ca_only (the default
            # returned by parse_cif even for no_ca / no_atom_site cases).
            if p.errors and any(
                err in {"nonfinite", "no_ca_atoms", "no_atom_site_columns", "bad_filename"}
                for err in p.errors
            ):
                continue
            modes.add(p.mode)
    if modes == {"full_backbone"}:
        mode = "full_backbone"
    elif modes == {"ca_only"}:
        mode = "ca_only"
    elif modes:
        mode = "mixed"
    else:
        mode = "empty"

    per_problem: dict = {}
    problem_scores: list = []
    for pid, parsed_list in parsed_by_problem.items():
        expected_len = expected_problems.get(pid, 0)
        if not parsed_list:
            per_problem[pid] = {"score": 0.0, "reason": "no_conformers"}
            problem_scores.append(0.0)
            continue

        n_conf = len(parsed_list)
        finite_frac = sum(
            1 for p in parsed_list if not any("nonfinite" in e for e in p.errors)
        ) / n_conf

        rg_values = [radius_of_gyration(p.ca_coords) for p in parsed_list if p.ca_coords.size]
        rg_mean = float(np.mean(rg_values)) if rg_values else 0.0
        n_res = expected_len or (parsed_list[0].residue_count if parsed_list else 0)
        rg_sc = _rg_score(rg_mean, n_res)

        spacing_vals = [ca_spacing_stats(p.ca_coords) for p in parsed_list if p.ca_coords.size]
        spacing_sc = (
            float(np.mean([_spacing_score(s) for s in spacing_vals])) if spacing_vals else 0.0
        )

        clash_vals = [
            ca_nonadjacent_min_distance(p.ca_coords) for p in parsed_list if p.ca_coords.size
        ]
        for idx, clash in enumerate(clash_vals, start=1):
            if math.isfinite(clash) and 2.0 <= clash < 3.0:
                soft_geometry_violations.append(f"{pid}:conf{idx}:soft_clash_{clash:.2f}")
        clash_sc = (
            float(np.mean([_clash_score(c) for c in clash_vals])) if clash_vals else 0.0
        )

        dihedral_vals = [
            pseudo_dihedral_smoothness(p.ca_coords) for p in parsed_list if p.ca_coords.size
        ]
        dihedral_sc = float(np.mean(dihedral_vals)) if dihedral_vals else 0.0

        traces = [p.ca_coords for p in parsed_list if p.ca_coords.size]
        if len(traces) >= 2:
            pairwise = ensemble_pairwise_matrix(traces)
            triu = pairwise[np.triu_indices_from(pairwise, k=1)]
            diversity = float(np.mean(triu)) if triu.size else 0.0
            rg_upper = 2.8 * max(n_res, 1) ** 0.52
            diversity_target = max(3.0, min(20.0, 0.35 * rg_upper))
            diversity_sc = _bounded_peak(diversity, 0.5, diversity_target, diversity_target * 3.0)
            pe = pca_effective_rank(traces)
            pca_sc = _pca_score(pe, len(traces))
            precision_penalty = medoid_outlier_ratio(pairwise)
            precision_sc = max(0.0, 1.0 - precision_penalty)
        else:
            diversity = 0.0
            diversity_sc = 0.0
            pca_sc = 0.0
            precision_sc = 0.5  # single-conformer: neutral

        unique_fraction, duplicate_penalty = _conformer_uniqueness(parsed_list)
        if n_conf > 1 and unique_fraction < 1.0:
            soft_geometry_violations.append(
                f"{pid}:duplicate_conformers:{unique_fraction:.3f}_unique_fraction"
            )

        problem_score = (
            W_DIVERSITY * diversity_sc
            + W_RG * rg_sc
            + W_SPACING * spacing_sc
            + W_CLASH * clash_sc
            + W_DIHEDRAL * dihedral_sc
            + W_PCA * pca_sc
            + W_PRECISION * precision_sc
            + W_FINITE * finite_frac
        )
        problem_score *= _soft_clash_penalty(clash_vals)
        problem_score *= duplicate_penalty
        per_problem[pid] = {
            "score": round(problem_score, 6),
            "diversity": round(diversity, 4),
            "rg_mean": round(rg_mean, 3),
            "clash_min": round(min(clash_vals), 3) if clash_vals else 0.0,
            "dihedral_smoothness": round(dihedral_sc, 4),
            "pca_rank": round(pca_sc, 4),
            "precision_proxy": round(precision_sc, 4),
            "finite_frac": round(finite_frac, 4),
            "soft_clash_penalty": round(_soft_clash_penalty(clash_vals), 4),
            "duplicate_penalty": round(duplicate_penalty, 4),
            "unique_conformer_fraction": round(unique_fraction, 4),
            "n_conformers": n_conf,
            "residue_count": parsed_list[0].residue_count if parsed_list else 0,
        }
        problem_scores.append(problem_score)

    format_violations, hard_geometry_violations = _split_violations(list(violations))
    geometry_violations = hard_geometry_violations + tuple(soft_geometry_violations)
    blocking_violations = format_violations + hard_geometry_violations
    final_score = 0.0 if blocking_violations else (
        float(np.mean(problem_scores)) if problem_scores else 0.0
    )

    return ProxyReport(
        score=round(final_score, 6),
        per_problem=per_problem,
        hard_gate_violations=blocking_violations,
        mode=mode,
        format_violations=format_violations,
        geometry_violations=geometry_violations,
    )
