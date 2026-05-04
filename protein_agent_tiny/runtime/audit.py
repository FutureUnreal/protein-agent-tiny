from __future__ import annotations

import json
import time
from pathlib import Path


def append_event(agent_log: Path, event_type: str, **payload) -> None:
    """Write a JSONL event. Creates parent dirs if needed."""
    agent_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"event_type": event_type, "timestamp_unix": int(time.time()), **payload}
    with agent_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
