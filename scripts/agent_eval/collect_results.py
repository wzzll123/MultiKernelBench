#!/usr/bin/env python3
"""Collect agent outputs and run external MultiKernelBench evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent outputs from workspaces")
    parser.add_argument("--workspace-root", type=Path, default=Path("experiment/tasks"))
    parser.add_argument("--language", type=str, required=True, help="Backend language for eval_single_runner")
    parser.add_argument("--output-filename", type=str, default="final_response.txt", help="Agent output txt filename")
    parser.add_argument("--out-json", type=Path, default=Path("experiment/outputs/agent_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiment/outputs/agent_summary.csv"))
    parser.add_argument("--eval-timeout", type=int, default=300, help="Timeout per task evaluation (seconds)")
    return parser.parse_args()


def evaluate_single(task_dir: Path, op: str, language: str, output_filename: str, timeout_sec: int) -> dict:
    output_path = task_dir / output_filename
    if not output_path.exists():
        return {
            "task": op,
            "status": "missing_output",
            "compiled": False,
            "correctness": False,
            "performance": None,
            "returncode": None,
            "error": f"{output_filename} not found",
        }

    tmp_result_path = task_dir / "raw_eval_result.json"
    cmd = [
        sys.executable,
        "eval_single_runner.py",
        str(output_path),
        op,
        language,
        str(tmp_result_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "task": op,
            "status": "eval_timeout",
            "compiled": False,
            "correctness": False,
            "performance": None,
            "returncode": None,
            "error": "evaluation timeout",
        }

    if tmp_result_path.exists():
        raw = json.loads(tmp_result_path.read_text())
        tmp_result_path.unlink(missing_ok=True)
    else:
        raw = {
            "compiled": False,
            "correctness": False,
            "performance": None,
            "compile_info": "raw eval result missing",
        }

    compiled = bool(raw.get("compiled", False))
    correctness = bool(raw.get("correctness", False))

    if not compiled:
        status = "compile_error"
    elif compiled and not correctness:
        status = "wrong_output"
    else:
        status = "passed"

    record = {
        "task": op,
        "status": status,
        "compiled": compiled,
        "correctness": correctness,
        "performance": raw.get("performance"),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "raw": raw,
        "error": None,
    }

    eval_result_path = task_dir / "eval_result.json"
    eval_result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    task_dirs = sorted([d for d in args.workspace_root.iterdir() if d.is_dir()], key=lambda p: p.name)

    rows = []
    for task_dir in task_dirs:
        row = evaluate_single(
            task_dir=task_dir,
            op=task_dir.name,
            language=args.language,
            output_filename=args.output_filename,
            timeout_sec=args.eval_timeout,
        )
        perf = row.get("performance") or {}
        row_csv = {
            "task": row["task"],
            "status": row["status"],
            "compiled": row["compiled"],
            "correctness": row["correctness"],
            "perf_mean": perf.get("mean"),
            "perf_std": perf.get("std"),
            "returncode": row["returncode"],
            "error": row.get("error"),
        }
        rows.append((row, row_csv))

    total = len(rows)
    compiled_cnt = sum(1 for r, _ in rows if r.get("compiled") is True)
    correct_cnt = sum(1 for r, _ in rows if r.get("correctness") is True)

    summary = {
        "workspace_root": str(args.workspace_root),
        "language": args.language,
        "output_filename": args.output_filename,
        "total_tasks": total,
        "compiled_tasks": compiled_cnt,
        "correct_tasks": correct_cnt,
        "compile_rate": compiled_cnt / total if total else 0.0,
        "correct_rate": correct_cnt / total if total else 0.0,
        "rows": [r for r, _ in rows],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "task",
        "status",
        "compiled",
        "correctness",
        "perf_mean",
        "perf_std",
        "returncode",
        "error",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row_csv in rows:
            writer.writerow(row_csv)

    print(f"[INFO] Wrote JSON summary to {args.out_json}")
    print(f"[INFO] Wrote CSV summary to {args.out_csv}")
    print(f"[INFO] compile_rate={summary['compile_rate']:.3f}, correct_rate={summary['correct_rate']:.3f}")


if __name__ == "__main__":
    main()
