#!/usr/bin/env python3
"""Prepare per-op isolated workspaces for directory-style agent evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset
from utils.utils import get_ref_src_path

TASK_MD_TEMPLATE = """# Kernel Optimization Task: {op}

You are working in a single isolated workspace.

## Goal
Produce a correct and efficient implementation in `candidate.py`.

## Files
- `reference.py`: reference implementation (read-only)
- `candidate.py`: your solution (editable)
- `run_eval.sh`: evaluation entrypoint
- `run_eval.py`: evaluation wrapper (do not modify)
- `result.json`: latest evaluation result
- `best_result.json`: best result seen in this workspace

## Rules
- Only work inside this directory.
- Do not modify `reference.py`, `run_eval.sh`, or `run_eval.py`.
- Do not inspect other tasks or parent directories.
- Do not bypass evaluation.
- Stop when the solution is correct and further improvements are unlikely.

## Evaluation
Run:
```bash
bash run_eval.sh
```

After each run, inspect `result.json`.

## Optimization priorities
1. correctness
2. stable execution
3. performance
"""

RUN_EVAL_PY = """#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__ROOT_DIR__)
OP = __OP__
LANGUAGE = __LANGUAGE__
TIMEOUT_SEC = __TIMEOUT_SEC__

RESULT_PATH = Path("result.json")
BEST_RESULT_PATH = Path("best_result.json")
BEST_CANDIDATE_PATH = Path("best_candidate.py")
TRAJECTORY_PATH = Path("trajectory.log")


def parse_result(raw):
    result = {
        "status": "failed",
        "compiled": False,
        "correctness": False,
        "performance": None,
        "error_type": "unknown",
        "raw": raw,
    }

    compiled = bool(raw.get("compiled", False))
    correctness = bool(raw.get("correctness", False))
    performance = raw.get("performance")

    result["compiled"] = compiled
    result["correctness"] = correctness
    result["performance"] = performance

    if not compiled:
        result["status"] = "compile_error"
        result["error_type"] = "compile_error"
    elif compiled and not correctness:
        result["status"] = "wrong_output"
        result["error_type"] = "wrong_output_or_runtime"
    elif correctness:
        result["status"] = "passed"
        result["error_type"] = None
    return result


def maybe_update_best(result):
    if not result.get("correctness", False):
        return False

    if not BEST_RESULT_PATH.exists():
        BEST_RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        shutil.copyfile("candidate.py", BEST_CANDIDATE_PATH)
        return True

    old = json.loads(BEST_RESULT_PATH.read_text())
    old_perf = ((old.get("performance") or {}).get("mean"))
    new_perf = ((result.get("performance") or {}).get("mean"))

    should_replace = old_perf is None or (new_perf is not None and new_perf < old_perf)
    if should_replace:
        BEST_RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        shutil.copyfile("candidate.py", BEST_CANDIDATE_PATH)
        return True
    return False


def write_trajectory(event):
    with TRAJECTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\\n")


def main():
    run_ts = datetime.now(timezone.utc).isoformat()
    if not Path("candidate.py").exists():
        RESULT_PATH.write_text(json.dumps({
            "status": "missing_candidate",
            "compiled": False,
            "correctness": False,
            "error_type": "missing_candidate",
        }, indent=2, ensure_ascii=False))
        return

    cmd = [
        sys.executable,
        str(ROOT_DIR / "eval_single_runner.py"),
        "candidate.py",
        OP,
        LANGUAGE,
        "raw_result.json",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=".",
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        timeout_result = {
            "status": "timeout",
            "compiled": False,
            "correctness": False,
            "performance": None,
            "error_type": "timeout",
        }
        RESULT_PATH.write_text(json.dumps(timeout_result, indent=2, ensure_ascii=False))
        write_trajectory({
            "timestamp": run_ts,
            "event": "eval_timeout",
            "timeout_sec": TIMEOUT_SEC,
        })
        return

    raw_path = Path("raw_result.json")
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        raw_path.unlink(missing_ok=True)
    else:
        raw = {
            "compiled": False,
            "correctness": False,
            "performance": None,
            "compile_info": "raw_result.json missing",
        }

    result = parse_result(raw)
    result["op"] = OP
    result["language"] = LANGUAGE
    result["returncode"] = proc.returncode
    result["stdout_tail"] = (proc.stdout or "")[-2000:]
    result["stderr_tail"] = (proc.stderr or "")[-2000:]
    result["raw"] = raw

    updated_best = maybe_update_best(result)
    result["updated_best"] = updated_best

    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_trajectory({
        "timestamp": run_ts,
        "event": "eval_finished",
        "status": result["status"],
        "compiled": result["compiled"],
        "correctness": result["correctness"],
        "updated_best": updated_best,
    })


if __name__ == "__main__":
    main()
"""

CANDIDATE_TEMPLATE = """from reference import Model


class ModelNew(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_init_inputs():
    return []


def get_inputs():
    return []
"""

RUN_EVAL_SH = """#!/usr/bin/env bash
set -euo pipefail

python run_eval.py > session.log 2>&1 || true
cat result.json
"""

AGENT_PROMPT = """You are an autonomous kernel optimization agent in an isolated workspace.

Workflow:
1) Read task.md and reference.py.
2) Edit only candidate.py.
3) Run `bash run_eval.sh`.
4) Read result.json.
5) Iterate until done.

When stopping, write a concise summary to agent_notes.md.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare isolated workspaces for agent evaluation")
    parser.add_argument("--language", type=str, default="cuda", help="Backend language")
    parser.add_argument("--categories", nargs="+", default=["activation"], help="Dataset categories or all")
    parser.add_argument("--ops", nargs="*", default=None, help="Optional explicit op list")
    parser.add_argument("--workspace-root", type=Path, default=Path("experiment/tasks"), help="Workspace root")
    parser.add_argument("--clean", action="store_true", help="Delete workspace root before prepare")
    parser.add_argument("--eval-timeout", type=int, default=300, help="run_eval.py timeout seconds")
    parser.add_argument("--readonly", action="store_true", help="Mark reference/eval/task files read-only")
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


def set_readonly(path: Path) -> None:
    os.chmod(path, 0o444)


def write_workspace(root: Path, op: str, language: str, eval_timeout: int, readonly: bool) -> None:
    task_dir = root / op
    task_dir.mkdir(parents=True, exist_ok=True)

    ref_src_path = Path(get_ref_src_path(op))
    shutil.copyfile(ref_src_path, task_dir / "reference.py")

    (task_dir / "candidate.py").write_text(CANDIDATE_TEMPLATE, encoding="utf-8")
    (task_dir / "task.md").write_text(TASK_MD_TEMPLATE.format(op=op), encoding="utf-8")
    (task_dir / "run_eval.py").write_text(
        dedent(RUN_EVAL_PY)
        .replace("__ROOT_DIR__", repr(str(Path.cwd())))
        .replace("__OP__", repr(op))
        .replace("__LANGUAGE__", repr(language))
        .replace("__TIMEOUT_SEC__", str(eval_timeout)),
        encoding="utf-8",
    )
    (task_dir / "run_eval.sh").write_text(RUN_EVAL_SH, encoding="utf-8")
    (task_dir / "result.json").write_text(json.dumps({"status": "not_run"}, indent=2), encoding="utf-8")
    (task_dir / "agent_notes.md").write_text("", encoding="utf-8")
    (task_dir / "session.log").write_text("", encoding="utf-8")
    (task_dir / "trajectory.log").write_text("", encoding="utf-8")
    (task_dir / "agent_prompt.txt").write_text(AGENT_PROMPT, encoding="utf-8")

    os.chmod(task_dir / "run_eval.sh", 0o755)
    os.chmod(task_dir / "run_eval.py", 0o755)

    if readonly:
        for name in ["reference.py", "run_eval.sh", "run_eval.py", "task.md", "agent_prompt.txt"]:
            set_readonly(task_dir / name)
        os.chmod(task_dir / "candidate.py", 0o644)


def main() -> None:
    args = parse_args()
    ops = pick_ops(args.categories, args.ops)

    workspace_root = args.workspace_root
    if args.clean and workspace_root.exists():
        shutil.rmtree(workspace_root)

    workspace_root.mkdir(parents=True, exist_ok=True)

    for op in ops:
        write_workspace(workspace_root, op, args.language, args.eval_timeout, args.readonly)

    manifest = {
        "language": args.language,
        "categories": args.categories,
        "ops": ops,
        "workspace_root": str(workspace_root),
        "eval_timeout": args.eval_timeout,
    }
    (workspace_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Prepared {len(ops)} workspaces at {workspace_root}")


if __name__ == "__main__":
    main()
