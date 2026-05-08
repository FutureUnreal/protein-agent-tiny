from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RuntimeConfig:
    iterations: int
    solver_rounds: int
    max_minutes: int
    workspace: Path
    output_dir: Path
    skip_agent: bool
    model: str
    base_url: Optional[str]
    bootstrap_max_attempts: int = 2


@dataclass(frozen=True)
class BootstrapResult:
    success: bool
    attempts: int
    sentinel_written: bool
    emergency_invoked: bool
    error: Optional[str]


@dataclass(frozen=True)
class ProxyReport:
    score: float
    per_problem: dict
    hard_gate_violations: tuple
    mode: str
    format_violations: tuple = ()
    geometry_violations: tuple = ()


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    accepted: bool
    score: float
    solver_changed: bool
    dependency_changed: bool
    report_path: Optional[str]
    run_id: Optional[str]
    stop_reason: Optional[str]
    metrics: object
    events_path: Optional[str]
    final_answer: str
    artifact_validation: object
    research_plan: str
    hypothesis: str
    observation: str
    reflection_run_id: Optional[str] = None
    reflection_stop_reason: Optional[str] = None
    reflection_metrics: object = None
    reflection_events_path: Optional[str] = None
    warnings: tuple = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    payload: dict
