#!/usr/bin/env python3
"""Prepare minimal per-op workspaces for directory-style agent generation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset
from utils.utils import get_ref_src_path

AGENT_PROMPT_TEMPLATE = """You are a kernel generation agent working in an isolated task directory.

Task: {op}

Files provided:
- reference.py (read-only)
- agent_prompt.txt

Requirements:
1. Read reference.py to understand the operator and interfaces.
2. Generate a candidate kernel solution.
3. Write your FINAL answer to `final_response.txt` in MultiKernelBench output format:
   - plain text response is allowed
   - or a fenced code block (```python ... ``` / ```cpp ... ```), which MultiKernelBench can parse
4. Do not write any other files.

Important:
- You are NOT allowed to self-evaluate in this directory.
- Do not access parent directories or other task directories.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare isolated workspaces for agent generation")
    parser.add_argument("--language", type=str, default="cuda", help="Backend language used later in external evaluation")
    parser.add_argument("--categories", nargs="+", default=["activation"], help="Dataset categories or all")
    parser.add_argument("--ops", nargs="*", default=None, help="Optional explicit op list")
    parser.add_argument("--workspace-root", type=Path, default=Path("experiment/tasks"), help="Workspace root")
    parser.add_argument("--clean", action="store_true", help="Delete workspace root before prepare")
    parser.add_argument("--readonly", action="store_true", help="Mark reference.py and agent_prompt.txt as read-only")
    return parser.parse_args()


def pick_ops(categories: list[str], explicit_ops: list[str] | None) -> list[str]:
    if explicit_ops:
        unknown = [op for op in explicit_ops if op not in dataset]
        if unknown:
            raise ValueError(f"Unknown ops: {unknown}")
        return explicit_ops

    ops = list(dataset.keys())
    if categories != ["all"]:
        ops = [op for op in ops if dataset[op]["category"] in categories]
    return ops


def write_workspace(root: Path, op: str, readonly: bool) -> None:
    task_dir = root / op
    task_dir.mkdir(parents=True, exist_ok=True)

    ref_src_path = Path(get_ref_src_path(op))
    shutil.copyfile(ref_src_path, task_dir / "reference.py")
    (task_dir / "agent_prompt.txt").write_text(AGENT_PROMPT_TEMPLATE.format(op=op), encoding="utf-8")

    if readonly:
        os.chmod(task_dir / "reference.py", 0o444)
        os.chmod(task_dir / "agent_prompt.txt", 0o444)


def main() -> None:
    args = parse_args()
    ops = pick_ops(args.categories, args.ops)

    workspace_root = args.workspace_root
    if args.clean and workspace_root.exists():
        shutil.rmtree(workspace_root)

    workspace_root.mkdir(parents=True, exist_ok=True)

    for op in ops:
        write_workspace(workspace_root, op, args.readonly)

    manifest = {
        "language": args.language,
        "categories": args.categories,
        "ops": ops,
        "workspace_root": str(workspace_root),
        "expected_agent_output_file": "final_response.txt",
        "notes": "Only reference.py + agent_prompt.txt are pre-created in each task directory.",
    }
    (workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Prepared {len(ops)} minimal workspaces at {workspace_root}")


if __name__ == "__main__":
    main()
