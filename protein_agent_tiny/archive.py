from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def archive_output(run_dir: Path, archive_root: Path | None = None, label: str | None = None) -> Path:
    run_dir = run_dir.resolve()
    root = run_dir.parent.parent if run_dir.parent.name == "outputs" else run_dir.parent
    archive_root = archive_root or root / "outputs" / "archive"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    archive_dir = archive_root / f"{stamp}{suffix}"
    archive_dir.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    for name in ("output.zip", "run_report.json", "technical_report.md"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)
            copied.append(name)

    agent_log = run_dir / "submission" / "agent.log"
    if agent_log.exists():
        shutil.copy2(agent_log, archive_dir / "agent.log")
        copied.append("agent.log")

    manifest = {
        "timestamp_unix": int(time.time()),
        "source_run_dir": str(run_dir),
        "archive_dir": str(archive_dir),
        "copied": copied,
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return archive_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/latest")
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    archive_dir = archive_output(
        Path(args.run_dir),
        Path(args.archive_root) if args.archive_root else None,
        args.label,
    )
    print(archive_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
