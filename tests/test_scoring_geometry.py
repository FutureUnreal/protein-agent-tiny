import numpy as np
import pytest

from protein_agent_tiny.scoring.geometry import (
    ca_nonadjacent_min_distance,
    ca_spacing_stats,
    ensemble_pairwise_matrix,
    medoid_outlier_ratio,
    pairwise_aligned_rmsd,
    pca_effective_rank,
    pseudo_dihedral_smoothness,
    radius_of_gyration,
)


def _uniform_line(n: int, spacing: float = 3.8) -> np.ndarray:
    """Straight-line CA trace with exact spacing."""
    ca = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ca[i, 0] = spacing * i
    return ca


def test_ca_spacing_stats_uniform():
    ca = _uniform_line(10)
    stats = ca_spacing_stats(ca)
    assert abs(stats["mean"] - 3.8) < 1e-9
    assert stats["std"] < 1e-9
    assert abs(stats["min"] - 3.8) < 1e-9
    assert abs(stats["max"] - 3.8) < 1e-9


def test_ca_spacing_stats_too_short():
    stats = ca_spacing_stats(np.zeros((1, 3)))
    assert stats["mean"] == 0.0


def test_pairwise_aligned_rmsd_identity():
    ca = _uniform_line(8)
    assert pairwise_aligned_rmsd(ca, ca) < 1e-9


def test_pairwise_aligned_rmsd_symmetric():
    rng = np.random.default_rng(42)
    a = rng.random((10, 3)) * 10.0
    b = rng.random((10, 3)) * 10.0
    assert abs(pairwise_aligned_rmsd(a, b) - pairwise_aligned_rmsd(b, a)) < 1e-9


def test_pca_effective_rank_duplicate_ensemble():
    ca = _uniform_line(10)
    traces = [ca.copy() for _ in range(5)]
    result = pca_effective_rank(traces)
    # All-duplicate ensemble: all variance is in 1 direction → rank ≈ 1
    assert result["rank"] < 1.5


def test_medoid_outlier_ratio_uniform_ensemble():
    ca = _uniform_line(10)
    traces = [ca.copy() for _ in range(5)]
    mat = ensemble_pairwise_matrix(traces)
    ratio = medoid_outlier_ratio(mat)
    assert ratio == 0.0


def test_medoid_outlier_ratio_range():
    rng = np.random.default_rng(7)
    n = 6
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = float(rng.random())
            mat[i, j] = v
            mat[j, i] = v
    ratio = medoid_outlier_ratio(mat)
    assert 0.0 <= ratio <= 1.0


def test_radius_of_gyration_nonzero():
    ca = _uniform_line(10)
    rg = radius_of_gyration(ca)
    assert rg > 0.0


def test_ca_nonadjacent_min_distance_no_clash():
    ca = _uniform_line(10)
    min_d = ca_nonadjacent_min_distance(ca)
    # Non-adjacent min dist in a straight line with 3.8Å spacing is 2*3.8=7.6
    assert min_d > 7.0


def test_pseudo_dihedral_smoothness_uniform():
    # Perfectly smooth (straight line) should return 1.0
    ca = _uniform_line(10)
    score = pseudo_dihedral_smoothness(ca)
    assert score == pytest.approx(1.0, abs=0.01)
