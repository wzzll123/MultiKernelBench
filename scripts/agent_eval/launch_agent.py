#!/usr/bin/env python3
"""Launch directory-style coding agents on prepared minimal workspaces."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent command for each workspace")
    parser.add_argument("--workspace-root", type=Path, default=Path("experiment/tasks"))
    parser.add_argument(
        "--agent-cmd",
        type=str,
        required=True,
        help=(
            "Agent command template. Use {prompt_file}, {prompt}, {workdir}. "
            "Example: qwen -p \"{prompt}\""
        ),
    )
    parser.add_argument("--timeout", type=int, default=1200, help="Per-task timeout seconds")
    parser.add_argument("--output-filename", type=str, default="final_response.txt", help="Expected agent output txt filename")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on number of tasks")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional explicit task ids")
    return parser.parse_args()


def load_manifest(workspace_root: Path) -> dict:
    manifest_path = workspace_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def pick_task_dirs(workspace_root: Path, task_filter: list[str] | None, max_tasks: int | None) -> list[Path]:
    task_dirs = [d for d in workspace_root.iterdir() if d.is_dir()]
    task_dirs.sort(key=lambda p: p.name)
    if task_filter:
        allow = set(task_filter)
        task_dirs = [d for d in task_dirs if d.name in allow]
    if max_tasks is not None:
        task_dirs = task_dirs[:max_tasks]
    return task_dirs


def run_single_task(task_dir: Path, command_template: str, timeout: int, output_filename: str) -> dict:
    prompt_file = task_dir / "agent_prompt.txt"
    prompt_text = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    command = command_template.format(
        prompt_file=str(prompt_file),
        prompt=prompt_text.replace('"', '\\"').replace("\n", " "),
        workdir=str(task_dir),
    )

    started_at = time.time()
    status = "finished"
    error = None

    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        returncode = None
        stdout_tail = ((e.stdout or "") if isinstance(e.stdout, str) else "")[-2000:]
        stderr_tail = ((e.stderr or "") if isinstance(e.stderr, str) else "")[-2000:]
    except Exception as e:  # noqa: BLE001
        status = "launch_error"
        returncode = None
        stdout_tail = ""
        stderr_tail = ""
        error = str(e)

    finished_at = time.time()
    output_exists = (task_dir / output_filename).exists()
    return {
        "task": task_dir.name,
        "status": status,
        "returncode": returncode,
        "duration_sec": round(finished_at - started_at, 3),
        "command": command,
        "output_filename": output_filename,
        "output_exists": output_exists,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.workspace_root)
    task_dirs = pick_task_dirs(args.workspace_root, args.task_filter, args.max_tasks)

    if not task_dirs:
        raise RuntimeError(f"No tasks found under {args.workspace_root}")

    run_records = []
    for task_dir in task_dirs:
        print(f"[INFO] Running agent on {task_dir.name}")
        record = run_single_task(task_dir, args.agent_cmd, args.timeout, args.output_filename)
        run_records.append(record)

        run_record_path = task_dir / "agent_run.json"
        run_record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "workspace_root": str(args.workspace_root),
        "manifest": manifest,
        "output_filename": args.output_filename,
        "num_tasks": len(task_dirs),
        "runs": run_records,
    }
    out_path = args.workspace_root / "agent_launch_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Agent launch summary: {out_path}")


if __name__ == "__main__":
    main()
