from pathlib import Path


category2exampleop = {
    "matmul": "matmul_add",
    "activation": "leaky_relu",
    "loss": "mse_loss",
    "normalization": "layer_norm",
    "reduce": "reduce_sum",
}


def _build_dataset():
    reference_root = Path(__file__).resolve().parent / "reference"
    tasks = {}

    for source_path in sorted(reference_root.glob("*/*.py")):
        if source_path.name.startswith("_"):
            continue

        category = source_path.parent.name
        task = {"category": category}
        if category.startswith("npukernelbench_level"):
            task["source"] = "NPUKernelBench"
            category2exampleop.setdefault(category, "add")
        tasks[source_path.stem] = task

    return tasks


dataset = _build_dataset()
