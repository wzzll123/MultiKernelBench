# Agent Evaluation Workflow (Directory-style)

This folder provides a complete directory-workflow pipeline to evaluate coding agents
(e.g., opencode/claudecode/qwen code) on MultiKernelBench tasks.

## What this workflow does

1. Creates isolated per-op workspaces.
2. Lets an external coding agent iterate on `candidate.py` using `bash run_eval.sh`.
3. Stores per-task outputs (`result.json`, `best_result.json`, logs, trajectory).
4. Aggregates all results into CSV/JSON summary.

## 1) Prepare workspaces

```bash
python scripts/agent_eval/prepare_workspaces.py \
  --language cuda \
  --categories activation \
  --workspace-root experiment/tasks \
  --clean \
  --readonly
```

Useful options:
- `--ops relu gelu`: prepare explicit ops.
- `--eval-timeout 300`: timeout for each `run_eval.py` execution.

## 2) Launch your agent

`launch_agent.py` receives a command template via `--agent-cmd`.
The template can use:
- `{prompt_file}`: path to `agent_prompt.txt`
- `{prompt}`: inline prompt text (single line)
- `{workdir}`: task directory

Example (adapt to your CLI):

```bash
python scripts/agent_eval/launch_agent.py \
  --workspace-root experiment/tasks \
  --agent-cmd 'qwen -p "{prompt}"' \
  --timeout 1200 \
  --max-tasks 10
```

Each task directory gets `agent_run.json`.

## 3) Collect results

```bash
python scripts/agent_eval/collect_results.py \
  --workspace-root experiment/tasks \
  --out-json experiment/outputs/agent_summary.json \
  --out-csv experiment/outputs/agent_summary.csv
```

## Per-task file layout

After preparation, each task folder contains:

- `task.md`: task rules and workflow.
- `reference.py`: copied reference implementation.
- `candidate.py`: editable candidate for agent.
- `run_eval.sh`: fixed evaluation entrypoint.
- `run_eval.py`: wrapper around MultiKernelBench single-op evaluator.
- `result.json`: latest eval result.
- `best_result.json`: best successful result so far.
- `best_candidate.py`: candidate for best result.
- `agent_notes.md`: agent's own summary.
- `session.log`: latest eval command output.
- `trajectory.log`: append-only eval event log.
- `agent_prompt.txt`: default prompt injected into agent.

## Notes

- `run_eval.py` internally calls `eval_single_runner.py`, so it keeps compile/correctness/performance behavior aligned with the existing benchmark pipeline.
- This workflow is deliberately metric-light and protocol-heavy. You can add more advanced metrics later without changing workspace semantics.
