import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "protein_agent_tiny"


def _imports(py_file: Path) -> set:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            if node.level > 0:
                imports.add("." * node.level + mod.split(".")[0])
            else:
                imports.add(mod.split(".")[0])
    return imports


def test_scoring_does_not_import_runtime_or_llm():
    forbidden = {"runtime", "all_in_agents", "memory", "literature", "environment"}
    forbidden_relative = {"..runtime", "..memory", "..literature", "..environment"}
    for py in (PKG / "scoring").glob("*.py"):
        imports = _imports(py)
        bad = imports & (forbidden | forbidden_relative)
        assert not bad, f"{py.name} has forbidden imports: {bad}"


def test_prompts_dir_has_no_python_logic():
    py_files = list((PKG / "prompts").glob("*.py"))
    assert len(py_files) == 1 and py_files[0].name == "__init__.py", \
        f"prompts/ should only have __init__.py, found: {[p.name for p in py_files]}"


def test_agent_runner_imports_only_runtime():
    allowed_top = {"__future__", "argparse", "os", "time", "pathlib", "."}
    imports = _imports(PKG / "agent_runner.py")
    extra = imports - allowed_top
    # Allow any '.' style relative import
    extra = {x for x in extra if not x.startswith(".")}
    assert not extra, f"agent_runner.py has unexpected top-level imports: {extra}"


def test_no_emergency_fallback_module():
    """The repo must not contain a fabricating emergency-fallback path.
    Failure honesty is enforced at the source-tree level: there is no
    geometric placeholder generator. If the agent cannot produce a real
    solver_pkg, the submission contains zero CIFs and the agent.log
    records the failure honestly."""
    assert not (PKG / "runtime" / "emergency_fallback.py").exists(), (
        "runtime/emergency_fallback.py must not exist — it would let the runtime "
        "fabricate placeholder CIFs to disguise agent failure."
    )


def test_runtime_imports_dag():
    """Ensure no module in runtime/ imports agent_runner (would create a cycle)."""
    for py in (PKG / "runtime").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "agent_runner" not in text, f"{py} imports agent_runner (would create cycle)"
