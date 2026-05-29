import argparse
import json
import re
from pathlib import Path


SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp"}
DEFAULT_MODEL_CANDIDATES = (
    "ModelNew.py",
    "model_new_ascendc.py",
    "model_new.py",
)
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "cmake-build-debug",
    "cmake-build-release",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Package an agent workdir project into a MultiKernelBench "
            "ascendc_direct_launch JSON submission."
        )
    )
    parser.add_argument(
        "project_dir",
        help="Project directory, for example cv_agent_workdir/agent_workdir/flash_attention.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path. Defaults to stdout.",
    )
    parser.add_argument(
        "--problem",
        help="Problem/operator name for the JSON metadata. Defaults to the project directory name.",
    )
    parser.add_argument(
        "--name",
        help="Submission name. Defaults to ascendc_direct_launch_<problem>.",
    )
    parser.add_argument(
        "--model",
        help=(
            "ModelNew source file relative to project_dir. Defaults to the first existing "
            "file among ModelNew.py, model_new_ascendc.py, and model_new.py."
        ),
    )
    parser.add_argument(
        "--entry",
        default="ModelNew.py::ModelNew",
        help="Model entry in <path>::<class> form. Defaults to ModelNew.py::ModelNew.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Defaults to 2.",
    )
    return parser.parse_args()


def sanitize_name(name):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return value or "project"


def resolve_model_path(project_dir, model_arg):
    if model_arg:
        model_path = project_dir / model_arg
        if not model_path.is_file():
            raise FileNotFoundError(f"Model source not found: {model_path}")
        return model_path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        model_path = project_dir / candidate
        if model_path.is_file():
            return model_path

    raise FileNotFoundError(
        "Cannot find ModelNew source. Pass --model or add one of: "
        + ", ".join(DEFAULT_MODEL_CANDIDATES)
    )


def iter_source_files(project_dir, model_path):
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(project_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if path == model_path:
            continue
        if path.suffix in SOURCE_SUFFIXES:
            yield path


def read_text(path):
    return path.read_text(encoding="utf-8")


def build_spec(project_dir, args):
    problem = sanitize_name(args.problem or project_dir.name)
    model_path = resolve_model_path(project_dir, args.model)

    sources = [
        {
            "path": "ModelNew.py",
            "content": read_text(model_path),
        }
    ]
    for source_path in iter_source_files(project_dir, model_path):
        sources.append(
            {
                "path": source_path.relative_to(project_dir).as_posix(),
                "content": read_text(source_path),
            }
        )

    return {
        "name": args.name or f"ascendc_direct_launch_{problem}",
        "author": "MultiKernelBench",
        "problem": problem,
        "format_version": "0.1",
        "description": (
            "JSON submission generated from an agent workdir project. "
            "It embeds ModelNew.py and Ascend C direct-launch kernel sources."
        ),
        "entry": {
            "model": args.entry,
        },
        "build": {
            "type": "ascendc_direct_launch",
        },
        "sources": sources,
    }


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    spec = build_spec(project_dir, args)
    output = json.dumps(spec, indent=args.indent, ensure_ascii=False) + "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
