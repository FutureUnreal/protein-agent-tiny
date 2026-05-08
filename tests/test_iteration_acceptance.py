from types import SimpleNamespace

from protein_agent_tiny.runtime.iteration import _accept_proxy


def _proxy(score, fmt=0, geom=0, mean=0.0, hard=()):
    return SimpleNamespace(
        score=score,
        format_violations=tuple(f"fmt{i}" for i in range(fmt)),
        geometry_violations=tuple(f"geom{i}" for i in range(geom)),
        hard_gate_violations=tuple(hard),
        per_problem={"1": {"score": mean}},
    )


def test_accepts_clean_score_non_regression():
    accepted, reason = _accept_proxy(_proxy(0.5), _proxy(0.4), 0.4)
    assert accepted
    assert reason == "score_nonregression_without_hard_gates"


def test_zero_score_accepts_ordered_geometry_improvement():
    best = _proxy(0.0, fmt=0, geom=4, mean=0.2, hard=("geom",))
    candidate = _proxy(0.0, fmt=0, geom=3, mean=0.2, hard=("geom",))
    accepted, reason = _accept_proxy(candidate, best, 0.0)
    assert accepted
    assert reason == "zero_score_geometry_violations_reduced"


def test_zero_score_accepts_format_fix_even_when_geometry_becomes_visible():
    best = _proxy(0.0, fmt=33, geom=0, mean=0.0, hard=tuple(f"fmt{i}" for i in range(33)))
    candidate = _proxy(0.0, fmt=0, geom=11, mean=0.0, hard=tuple(f"geom{i}" for i in range(11)))
    accepted, reason = _accept_proxy(candidate, best, 0.0)
    assert accepted
    assert reason == "zero_score_format_violations_reduced"


def test_zero_score_rejects_improvement_with_lower_mean():
    best = _proxy(0.0, fmt=0, geom=4, mean=0.2, hard=("geom",))
    candidate = _proxy(0.0, fmt=0, geom=3, mean=0.1, hard=("geom",))
    accepted, reason = _accept_proxy(candidate, best, 0.0)
    assert not accepted
    assert reason == "zero_score_no_ordered_improvement"


def test_zero_score_accepts_mean_improvement_when_counts_equal():
    best = _proxy(0.0, fmt=0, geom=4, mean=0.2, hard=("geom",))
    candidate = _proxy(0.0, fmt=0, geom=4, mean=0.25, hard=("geom",))
    accepted, reason = _accept_proxy(candidate, best, 0.0)
    assert accepted
    assert reason == "zero_score_mean_problem_score_improved"
