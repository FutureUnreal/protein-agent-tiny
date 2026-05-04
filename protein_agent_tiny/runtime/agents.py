from __future__ import annotations

import os
from pathlib import Path

from ..prompts import load
from ..tools import validate_submission_tool
from .contracts import RuntimeConfig


def _make_registry_and_policy(workspace: Path):
    from all_in_agents import BUILTIN_TOOLS, ToolPolicy, ToolRegistry, unsafe_defaults

    registry = ToolRegistry(approval_callback=unsafe_defaults())
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    registry.register(validate_submission_tool)
    # Tools execute subprocesses inside the workspace and may write or print env vars.
    # We sanitize so API keys (OPENAI_API_KEY, OPENALEX_API_KEY, etc.) loaded from
    # the top-level .env stay inside the LLM adapter and never reach tool stdout
    # or the agent log.
    policy = ToolPolicy(
        require_approval_for=frozenset(),
        workspace_roots=(workspace.resolve(),),
        command_denylist=frozenset({"rm", "del", "rmdir"}),
        sanitize_env=True,
    )
    return registry, policy


def _budget(max_tool_calls: int, max_llm_calls: int = 20):
    from all_in_agents import Budget

    return Budget(
        max_llm_calls=int(os.environ.get("PROTEIN_AGENT_MAX_LLM_CALLS", str(max_llm_calls))),
        max_tool_calls=int(os.environ.get("PROTEIN_AGENT_MAX_TOOL_CALLS", str(max_tool_calls))),
        max_wall_ms=int(os.environ.get("PROTEIN_AGENT_MAX_WALL_MS", "1800000")),
        max_input_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_INPUT_TOKENS", "128000")),
        max_output_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_OUTPUT_TOKENS", "8192")),
        loop_same_action_limit=4,
    )


def build_bootstrap_agent(cfg: RuntimeConfig, workspace: Path, run_dir: Path):
    """First-version creator. Requires cli.py + pipeline.py + sentinel."""
    from all_in_agents import Agent, ArtifactContract, ArtifactSpec, OpenAIAdapter

    registry, policy = _make_registry_and_policy(workspace)
    contract = ArtifactContract((
        ArtifactSpec("research_plan.md", min_bytes=200, description="Bootstrap research plan."),
        ArtifactSpec("hypothesis.md", min_bytes=80, description="Bootstrap hypothesis."),
        ArtifactSpec("notes.md", min_bytes=20, description="Bootstrap notes."),
        ArtifactSpec("solver_pkg/cli.py", min_bytes=200, description="CLI shim."),
        ArtifactSpec("solver_pkg/pipeline.py", min_bytes=200, description="Pipeline core."),
        ArtifactSpec("solver_pkg/.pipeline_ready", min_bytes=1, description="Sentinel."),
    ))
    return Agent(
        llm=OpenAIAdapter(model=cfg.model, base_url=cfg.base_url, max_retries=2),
        tools=registry,
        budget=_budget(max_tool_calls=120, max_llm_calls=30),
        run_dir=str(run_dir),
        system=load("bootstrap.md"),
        workspace_root=str(workspace),
        tool_policy=policy,
        project_root=str(workspace),
        skills=("protein-ensemble",),
        artifact_contract=contract,
    )


def build_improve_agent(cfg: RuntimeConfig, workspace: Path, run_dir: Path):
    """Evolves existing solver_pkg/. Requires cli.py stays present."""
    from all_in_agents import Agent, ArtifactContract, ArtifactSpec, OpenAIAdapter

    registry, policy = _make_registry_and_policy(workspace)
    contract = ArtifactContract((
        ArtifactSpec("research_plan.md", min_bytes=200, description="Iteration research plan."),
        ArtifactSpec("hypothesis.md", min_bytes=80, description="Iteration hypothesis."),
        ArtifactSpec("notes.md", min_bytes=20, description="Iteration notes."),
        ArtifactSpec("solver_pkg/cli.py", min_bytes=200, description="CLI shim must remain."),
    ))
    return Agent(
        llm=OpenAIAdapter(model=cfg.model, base_url=cfg.base_url, max_retries=2),
        tools=registry,
        budget=_budget(max_tool_calls=60, max_llm_calls=20),
        run_dir=str(run_dir),
        system=load("improve.md"),
        workspace_root=str(workspace),
        tool_policy=policy,
        project_root=str(workspace),
        skills=("protein-ensemble",),
        artifact_contract=contract,
    )


def build_reflect_agent(cfg: RuntimeConfig, workspace: Path, run_dir: Path):
    """Reflection phase: no tools, max 1 LLM call."""
    from all_in_agents import Agent, Budget, OpenAIAdapter, ToolPolicy, ToolRegistry

    budget = Budget(
        max_llm_calls=1,
        max_tool_calls=0,
        max_wall_ms=180000,
        max_input_tokens_per_call=int(os.environ.get("PROTEIN_AGENT_MAX_INPUT_TOKENS", "128000")),
        max_output_tokens_per_call=min(int(os.environ.get("PROTEIN_AGENT_MAX_OUTPUT_TOKENS", "8192")), 4096),
    )
    policy = ToolPolicy(
        require_approval_for=frozenset(),
        workspace_roots=(workspace.resolve(),),
        command_denylist=frozenset({"rm", "del", "rmdir"}),
        sanitize_env=True,
    )
    return Agent(
        llm=OpenAIAdapter(model=cfg.model, base_url=cfg.base_url, max_retries=2),
        tools=ToolRegistry(),
        budget=budget,
        run_dir=str(run_dir),
        system=load("reflect.md"),
        workspace_root=str(workspace),
        tool_policy=policy,
        project_root=str(workspace),
        skills=("protein-ensemble",),
    )
