from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .runtime.contracts import RuntimeConfig
from .runtime.iteration import run as runtime_run

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "outputs" / "latest"
DEFAULT_WORKSPACE = ROOT / "workspaces" / "current"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        prog="protein_agent_tiny.agent_runner",
        description="AI4S task 3 agent runner (bootstrap/improve/reflect state machine).",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--solver-rounds", type=int, default=1)
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument(
        "--workspace", default=None,
        help="Workspace directory. Defaults to workspaces/current so reruns can resume an existing solver_pkg via the sentinel fast-path.",
    )
    parser.add_argument(
        "--fresh-workspace", action="store_true",
        help="Create a timestamped workspace instead of reusing workspaces/current.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-agent", action="store_true",
                        help="Package-only audit mode: skip LLM path, write zero CIFs and an audit_only agent.log. Useful for CI; NOT a valid competition submission.")
    parser.add_argument("--bootstrap-max-attempts", type=int,
                        default=int(os.environ.get("PROTEIN_AGENT_BOOTSTRAP_MAX_ATTEMPTS", "2")))
    args = parser.parse_args()

    os.environ["PROTEIN_AGENT_MAX_WALL_MS"] = str(args.max_minutes * 60 * 1000)
    model = (
        os.environ.get("PROTEIN_AGENT_MODEL")
        or os.environ.get("DEFAULT_MODEL")
        or "step-3.5-flash-2603"
    )
    base_url = os.environ.get("OPENAI_API_BASE")

    if args.workspace:
        workspace = Path(args.workspace)
    elif args.fresh_workspace:
        workspace = ROOT / "workspaces" / time.strftime("%Y%m%d_%H%M%S")
    else:
        workspace = DEFAULT_WORKSPACE

    return RuntimeConfig(
        iterations=max(1, args.iterations),
        solver_rounds=max(1, args.solver_rounds),
        max_minutes=max(1, args.max_minutes),
        workspace=workspace,
        output_dir=Path(args.out),
        skip_agent=args.skip_agent,
        model=model,
        base_url=base_url,
        bootstrap_max_attempts=max(1, args.bootstrap_max_attempts),
    )


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = parse_args()
    return runtime_run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
