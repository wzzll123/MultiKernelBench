# Agent Evaluation Workflow (Minimal Directory Mode)

This workflow is the **minimal agent setting**:
- each task directory only pre-creates `reference.py` and `agent_prompt.txt`
- no in-directory self-evaluation scripts are provided to the agent
- the agent must write a single output text file: `final_response.txt`
- external controller evaluates those outputs using MultiKernelBench evaluator

## 1) Prepare minimal workspaces

```bash
python scripts/agent_eval/prepare_workspaces.py \
  --language cuda \
  --strategy add_shot \
  --categories activation \
  --workspace-root experiment/tasks \
  --clean \
  --readonly
```

After preparation, each task folder initially contains only:
- `reference.py`
- `agent_prompt.txt`

`agent_prompt.txt` is generated dynamically from `prompt_generators/{language}_{strategy}.py` (aligned with existing benchmark prompts), then appends only the directory-output requirement (`final_response.txt`).

## 2) Launch your agent

`launch_agent.py` supports placeholders in `--agent-cmd`:
- `{prompt_file}`
- `{prompt}`
- `{workdir}`

Example:

```bash
python scripts/agent_eval/launch_agent.py \
  --workspace-root experiment/tasks \
  --agent-cmd 'qwen -p "{prompt}"' \
  --timeout 1200 \
  --output-filename final_response.txt \
  --auto-yolo \
  --max-tasks 10
```

Expected: the agent writes `final_response.txt` in each task directory.

`--max-tasks` is mainly for quick smoke runs / cost control: it only runs the first N task directories (sorted by task name), so you can debug workflow on small subsets before full-scale runs.

## 3) Evaluate agent outputs externally

```bash
python scripts/agent_eval/collect_results.py \
  --workspace-root experiment/tasks \
  --language cuda \
  --output-filename final_response.txt \
  --out-json experiment/outputs/agent_summary.json \
  --out-csv experiment/outputs/agent_summary.csv
```

This calls `eval_single_runner.py` for each task output and writes:
- per-task `eval_result.json`
- aggregated JSON/CSV summary

## Notes

- Keep agent output compatible with MultiKernelBench model output style (plain response text or fenced code block).
- If evaluation environment lacks hardware/runtime (e.g., no CUDA), records may show compile/runtime failures due to environment limitations.


If you use Qwen CLI in non-interactive mode, `--auto-yolo` is recommended (it appends `-y` automatically when missing).

