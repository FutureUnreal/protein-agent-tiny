from __future__ import annotations

import json
from pathlib import Path

from all_in_agents import SideEffectLevel, Tool, ToolResponse

from .validate import validate_submission


async def _validate_submission_impl(args: dict, run) -> ToolResponse:
    path = Path(run.workspace_root or ".") / args["submission_dir"]
    result = validate_submission(path.resolve())
    return ToolResponse("success" if result["ok"] else "error", json.dumps(result, indent=2))


validate_submission_tool = Tool(
    name="validate_submission",
    description="Validate an AI4S protein ensemble submission directory.",
    input_schema={
        "type": "object",
        "properties": {
            "submission_dir": {"type": "string", "description": "Submission directory path relative to workspace"}
        },
        "required": ["submission_dir"],
    },
    side_effect_level=SideEffectLevel.READ_ONLY,
    execute=_validate_submission_impl,
)
