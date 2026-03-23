#!/usr/bin/env python3
"""Prepare minimal per-op workspaces for directory-style agent generation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset
from prompt_generators.prompt_registry import PROMPT_REGISTRY
from utils.utils import get_ref_src_path

OUTPUT_REQUIREMENT_SUFFIX = '''

# Additional execution constraints for directory workflow
- You are running in an isolated task directory.
- Do NOT modify any file except writing your final answer.
- Write your final answer to `final_response.txt` in this directory.
- Keep the same output style required above (code-only if requested, no extra commentary if requested).
- Do not self-evaluate or access other directories.
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare isolated workspaces for agent generation")
    parser.add_argument("--language", type=str, default="cuda", help="Prompt language/backend (must exist in prompt registry)")
    parser.add_argument("--strategy", type=str, default="add_shot", help="Prompt strategy (e.g. add_shot / selected_shot)")
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


def generate_prompt(language: str, strategy_name: str, op: str) -> str:
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        try:
            importlib.import_module(f"prompt_generators.{language}_{strategy_name}")
        except ImportError as e:
            raise ValueError(
                f"Unsupported prompt config: language={language}, strategy={strategy_name}"
            ) from e

    strategy = PROMPT_REGISTRY[language][strategy_name]
    return strategy.generate(op)


def build_agent_prompt(language: str, strategy: str, op: str) -> str:
    base_prompt = generate_prompt(language, strategy, op)
    return base_prompt.rstrip() + "\n" + OUTPUT_REQUIREMENT_SUFFIX


def write_workspace(root: Path, op: str, language: str, strategy: str, readonly: bool) -> None:
    task_dir = root / op
    task_dir.mkdir(parents=True, exist_ok=True)

    ref_src_path = Path(get_ref_src_path(op))
    shutil.copyfile(ref_src_path, task_dir / "reference.py")

    agent_prompt = build_agent_prompt(language, strategy, op)
    (task_dir / "agent_prompt.txt").write_text(agent_prompt, encoding="utf-8")

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
        write_workspace(workspace_root, op, args.language, args.strategy, args.readonly)

    manifest = {
        "language": args.language,
        "strategy": args.strategy,
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
