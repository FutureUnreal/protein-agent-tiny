from protein_agent_tiny.runtime.iteration import _forced_improve_mode


def test_forced_mode_triggers_dependency_experiment_after_three_failures():
    forced_mode, forced_directive = _forced_improve_mode(3)
    assert forced_mode["mode"] == "dependency experiment"
    assert forced_mode["streak"] == 3
    assert "ESMFold" in forced_directive or "AlphaFold2" in forced_directive


def test_forced_mode_is_empty_before_threshold():
    forced_mode, forced_directive = _forced_improve_mode(2)
    assert forced_mode == {}
    assert forced_directive == ""
