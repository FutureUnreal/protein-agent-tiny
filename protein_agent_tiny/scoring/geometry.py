from __future__ import annotations

import numpy as np

from .cif import ParsedCif  # noqa: F401 — type hints only


def ca_spacing_stats(ca: np.ndarray) -> dict:
    """Return {"min", "max", "mean", "std"} of adjacent CA-CA distances.

    If len(ca) < 2, all values are 0.0.
    """
    if len(ca) < 2:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    diffs = np.linalg.norm(np.diff(ca, axis=0), axis=1).astype(np.float64)
    return {
        "min": float(diffs.min()),
        "max": float(diffs.max()),
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
    }


def radius_of_gyration(ca: np.ndarray) -> float:
    """sqrt(mean(|x_i - centroid|^2))."""
    if len(ca) == 0:
        return 0.0
    centroid = ca.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((ca - centroid) ** 2, axis=1))))


def end_to_end(ca: np.ndarray) -> float:
    """|ca[-1] - ca[0]|."""
    if len(ca) < 2:
        return 0.0
    return float(np.linalg.norm(ca[-1] - ca[0]))


def pairwise_aligned_rmsd(ca_a: np.ndarray, ca_b: np.ndarray) -> float:
    """Kabsch-aligned CA-RMSD via SVD; no scipy.

    Trims both arrays to min(len_a, len_b). Centers each, computes rotation R
    via SVD of A.T @ B, handles reflection by flipping sign of last row of Vt
    if det(U @ Vt) < 0. Applies R to A, then computes RMSD.
    """
    n = min(len(ca_a), len(ca_b))
    if n == 0:
        return 0.0
    A = ca_a[:n].astype(np.float64).copy()
    B = ca_b[:n].astype(np.float64).copy()

    # Center both
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)

    # Covariance
    H = A.T @ B  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)

    # Correct reflection
    d = np.linalg.det(U @ Vt)
    diag = np.ones(3, dtype=np.float64)
    if d < 0:
        diag[-1] = -1.0
    R = Vt.T @ (diag[:, None] * U.T)

    A_rot = A @ R.T
    diff = A_rot - B
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def ensemble_pairwise_matrix(traces: list[np.ndarray]) -> np.ndarray:
    """Symmetric NxN matrix of pairwise_aligned_rmsd; diagonal = 0."""
    n = len(traces)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = pairwise_aligned_rmsd(traces[i], traces[j])
            mat[i, j] = v
            mat[j, i] = v
    return mat


def ca_nonadjacent_min_distance(ca: np.ndarray) -> float:
    """Clash proxy: minimum CA-CA distance excluding |i-j| < 2.

    Returns +inf if < 3 residues.
    """
    n = len(ca)
    if n < 3:
        return float("inf")
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 2, n):
            d = float(np.linalg.norm(ca[i] - ca[j]))
            if d < min_dist:
                min_dist = d
    return min_dist


def pseudo_dihedral_smoothness(ca: np.ndarray) -> float:
    """CA-based pseudo-torsion smoothness proxy for Ramachandran.

    For each quadruple (ca[i], ca[i+1], ca[i+2], ca[i+3]) compute the
    pseudo-dihedral angle. Returns 1 - normalized_variance of consecutive
    angle differences (higher = smoother). Returns 1.0 if < 4 residues.
    Output bounded to [0, 1].
    """
    if len(ca) < 4:
        return 1.0

    angles: list[float] = []
    for i in range(len(ca) - 3):
        b1 = ca[i + 1] - ca[i]
        b2 = ca[i + 2] - ca[i + 1]
        b3 = ca[i + 3] - ca[i + 2]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-12 or n2_norm < 1e-12:
            angles.append(0.0)
            continue

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle = np.arccos(cos_a)
        angles.append(float(angle))

    if len(angles) < 2:
        return 1.0

    angles_arr = np.array(angles, dtype=np.float64)
    diffs = np.diff(angles_arr)
    variance = float(np.var(diffs))
    # Normalize: max possible variance for angles in [0, pi] is (pi^2)/4
    max_var = (np.pi ** 2) / 4.0
    normalized_var = variance / max_var
    return float(np.clip(1.0 - normalized_var, 0.0, 1.0))


def pca_effective_rank(traces: list[np.ndarray]) -> dict:
    """Align each trace to the first via Kabsch, flatten coords (n_conf, 3*res).

    Center, SVD singular values s.
    effective_rank = exp(entropy(s^2 / sum(s^2))), entropy = -sum(p*log(p)).
    Returns {"rank": float, "entropy": float, "singular_values": list[float]}.
    If n_conf < 2 or residue count mismatch among traces, returns rank=1.0,
    entropy=0.0, singular_values=[].
    """
    _default = {"rank": 1.0, "entropy": 0.0, "singular_values": []}

    if len(traces) < 2:
        return _default

    n_res = len(traces[0])
    if n_res == 0:
        return _default

    # Check consistent residue counts
    for t in traces[1:]:
        if len(t) != n_res:
            return _default

    # Align each trace to first via Kabsch
    ref = traces[0].astype(np.float64).copy()
    ref_centered = ref - ref.mean(axis=0)

    aligned: list[np.ndarray] = [ref_centered.copy()]
    for t in traces[1:]:
        mob = t.astype(np.float64).copy()
        mob_centered = mob - mob.mean(axis=0)

        H = ref_centered.T @ mob_centered
        U, _S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U @ Vt)
        diag = np.ones(3, dtype=np.float64)
        if d < 0:
            diag[-1] = -1.0
        R = Vt.T @ (diag[:, None] * U.T)
        aligned.append((mob_centered @ R.T))

    # Build (n_conf, 3*n_res) matrix and center across conformers
    X = np.array([a.flatten() for a in aligned], dtype=np.float64)
    X -= X.mean(axis=0)

    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-30:
        return _default

    p = s2 / total
    # Avoid log(0): only sum where p > 0
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    rank = float(np.exp(entropy))

    return {"rank": rank, "entropy": entropy, "singular_values": s.tolist()}


def medoid_outlier_ratio(pairwise: np.ndarray, threshold_factor: float = 2.0) -> float:
    """Reverse-precision proxy.

    1. medoid = argmin_i sum_j pairwise[i,j]
    2. dists = pairwise[medoid, :]
    3. threshold = threshold_factor * median(dists[dists>0])
    4. ratio = fraction of conformers whose dist > threshold
    Returns value in [0, 1]. If n < 2, return 0.0.
    """
    n = len(pairwise)
    if n < 2:
        return 0.0

    row_sums = pairwise.sum(axis=1)
    medoid = int(np.argmin(row_sums))
    dists = pairwise[medoid, :].astype(np.float64)

    positive = dists[dists > 0]
    if len(positive) == 0:
        return 0.0

    threshold = threshold_factor * float(np.median(positive))
    ratio = float(np.sum(dists > threshold) / n)
    return float(np.clip(ratio, 0.0, 1.0))
