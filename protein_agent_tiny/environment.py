from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_probe(command: list[str], timeout: int = 8) -> dict[str, object]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:4000],
            "stderr": proc.stderr.strip()[:2000],
        }
    except FileNotFoundError:
        return {"available": False, "error": "not found"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def read_meminfo() -> dict[str, object]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    data: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if parts and parts[0].isdigit():
            data[key] = int(parts[0])
    return data


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def probe_environment(root: Path) -> dict[str, object]:
    disk = shutil.disk_usage(root)
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "nvcc": ["nvcc", "--version"],
        "rosetta": ["rosetta_scripts", "-help"],
        "foldseek": ["foldseek", "version"],
    }
    modules = ["torch", "numpy", "scipy", "openmm", "Bio", "biotite", "all_in_agents"]
    env = {
        "timestamp_unix": int(time.time()),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "memory_kb": read_meminfo(),
        "disk": {
            "path": str(root),
            "total_gb": round(disk.total / 1024**3, 2),
            "used_gb": round(disk.used / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
        },
        "commands": {name: run_probe(command) for name, command in commands.items()},
        "python_modules": {name: module_available(name) for name in modules},
    }
    torch = env["python_modules"].get("torch")
    if torch:
        env["torch_cuda"] = run_probe([
            sys.executable,
            "-c",
            "import torch, json; print(json.dumps({'available': torch.cuda.is_available(), 'count': torch.cuda.device_count(), 'names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))",
        ])
    return env


def render_environment_report(env: dict[str, object]) -> str:
    commands = env.get("commands", {})
    modules = env.get("python_modules", {})
    memory = env.get("memory_kb", {})
    mem_total_gb = None
    if isinstance(memory, dict) and isinstance(memory.get("MemTotal"), int):
        mem_total_gb = round(memory["MemTotal"] / 1024**2, 2)
    lines = [
        "# Environment Report",
        "",
        f"- Platform: `{env.get('platform')}`",
        f"- Python: `{env.get('executable')}`",
        f"- CPU count: `{env.get('cpu_count')}`",
        f"- Memory total GB: `{mem_total_gb}`",
        f"- Disk free GB: `{(env.get('disk') or {}).get('free_gb')}`",
        "",
        "## Commands",
        "",
    ]
    if isinstance(commands, dict):
        for name, result in commands.items():
            available = result.get("available") if isinstance(result, dict) else False
            lines.append(f"- `{name}`: `{available}`")
    lines.extend(["", "## Python Modules", ""])
    if isinstance(modules, dict):
        for name, available in modules.items():
            lines.append(f"- `{name}`: `{available}`")
    torch_cuda = env.get("torch_cuda")
    if torch_cuda:
        lines.extend(["", "## Torch CUDA", "", "```json", json.dumps(torch_cuda, indent=2), "```"])
    lines.append("")
    return "\n".join(lines)


def write_environment_report(workspace: Path, root: Path) -> dict[str, object]:
    env = probe_environment(root)
    (workspace / "environment_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (workspace / "environment_report.md").write_text(render_environment_report(env), encoding="utf-8")
    return env
