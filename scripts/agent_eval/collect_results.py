#!/usr/bin/env python3
"""Collect workspace results into CSV/JSON summary for agent evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect agent evaluation results")
    parser.add_argument("--workspace-root", type=Path, default=Path("experiment/tasks"))
    parser.add_argument("--out-json", type=Path, default=Path("experiment/outputs/agent_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiment/outputs/agent_summary.csv"))
    return parser.parse_args()


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    task_dirs = sorted([d for d in args.workspace_root.iterdir() if d.is_dir()], key=lambda p: p.name)

    rows = []
    for task_dir in task_dirs:
        result = read_json(task_dir / "result.json", default={"status": "missing"})
        best = read_json(task_dir / "best_result.json", default={})
        run = read_json(task_dir / "agent_run.json", default={})

        perf = result.get("performance") or {}
        best_perf = best.get("performance") or {}

        row = {
            "task": task_dir.name,
            "status": result.get("status"),
            "compiled": result.get("compiled"),
            "correctness": result.get("correctness"),
            "perf_mean": perf.get("mean"),
            "perf_std": perf.get("std"),
            "best_perf_mean": best_perf.get("mean"),
            "updated_best": result.get("updated_best"),
            "agent_status": run.get("status"),
            "agent_duration_sec": run.get("duration_sec"),
            "agent_returncode": run.get("returncode"),
        }
        rows.append(row)

    total = len(rows)
    compiled_cnt = sum(1 for r in rows if r.get("compiled") is True)
    correct_cnt = sum(1 for r in rows if r.get("correctness") is True)

    summary = {
        "workspace_root": str(args.workspace_root),
        "total_tasks": total,
        "compiled_tasks": compiled_cnt,
        "correct_tasks": correct_cnt,
        "compile_rate": compiled_cnt / total if total else 0.0,
        "correct_rate": correct_cnt / total if total else 0.0,
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else [
        "task",
        "status",
        "compiled",
        "correctness",
        "perf_mean",
        "perf_std",
        "best_perf_mean",
        "updated_best",
        "agent_status",
        "agent_duration_sec",
        "agent_returncode",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote JSON summary to {args.out_json}")
    print(f"[INFO] Wrote CSV summary to {args.out_csv}")
    print(f"[INFO] compile_rate={summary['compile_rate']:.3f}, correct_rate={summary['correct_rate']:.3f}")


if __name__ == "__main__":
    main()
