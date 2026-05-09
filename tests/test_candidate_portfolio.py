import json
from types import SimpleNamespace

from protein_agent_tiny.runtime.iteration import (
    _snapshot_candidate_pipeline,
    _write_iteration_context,
)


def test_candidate_snapshot_is_exposed_in_iteration_context(tmp_path):
    workspace = tmp_path
    solver_pkg = workspace / "solver_pkg"
    solver_pkg.mkdir()
    (solver_pkg / "pipeline.py").write_text("VALUE = 1\n", encoding="utf-8")

    proxy = SimpleNamespace(
        score=0.42,
        hard_gate_violations=(),
        format_violations=(),
        geometry_violations=("1:conf1:soft_clash_2.50",),
        per_problem={
            "1": {
                "score": 0.42,
                "generated_pairwise_rmsd": 0.7,
                "generated_effective_rank_score": 0.5,
                "medoid_outlier_score": 0.9,
            }
        },
        mode="ca_only",
    )

    _snapshot_candidate_pipeline(
        workspace,
        iteration=1,
        proxy=proxy,
        accepted=False,
        reason="test_rejection",
        score_dir=workspace / "iteration_runs" / "score_test",
        solver_changed=True,
    )
    _write_iteration_context(
        workspace,
        iteration=2,
        total=5,
        best_score=0.5,
        history=[],
        current_proxy=proxy,
    )

    ctx = json.loads((workspace / "iteration_context.json").read_text(encoding="utf-8"))
    assert ctx["candidate_portfolio"][0]["iteration"] == 1
    assert ctx["candidate_portfolio"][0]["selection_score_proxy"] == 0.42
    assert ctx["candidate_portfolio"][0]["reason"] == "test_rejection"
    assert (workspace / "candidate_pipelines" / "iter_01_0.420000_rejected" / "solver_pkg" / "pipeline.py").exists()
    assert "cannot estimate official coverage" in ctx["score_proxy_definition"]
