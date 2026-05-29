import argparse
import json
from pathlib import Path

from dataset import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one generated response for one MultiKernelBench task."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the generated response text file.",
    )
    parser.add_argument(
        "-o",
        "--op",
        dest="op",
        required=True,
        help="Task/operator name, such as relu or 1_logical_and.",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        required=True,
        help="Backend language name, such as cuda, ascendc, or ascendc_direct_launch.",
    )
    parser.add_argument(
        "-r",
        "--result",
        dest="result_path",
        required=True,
        help="Path where the JSON result should be written.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for the result file. Defaults to 2.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    result_path = Path(args.result_path)

    if args.op not in dataset:
        raise ValueError(f"Unknown op: {args.op}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Generated response file not found: {input_path}")

    from utils.evaluation_utils import eval_single

    response_txt = input_path.read_text(encoding="utf-8")
    result = eval_single(response_txt, args.op, args.language)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=args.indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
