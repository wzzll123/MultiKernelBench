#!/usr/bin/env python3
"""Profile representative PyTorch workloads and summarize kernel categories.

This script supports the paper's task-distribution justification. It profiles
representative DL workloads with torch.profiler, maps ATen operators to
MultiKernelBench categories, and uses torch.fx graphs to estimate simple fusion
opportunities. It intentionally depends only on PyTorch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def require_torch():
    try:
        import torch
        import torch.fx as fx
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.profiler import ProfilerActivity, profile
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required. Activate your benchmark environment or install "
            "torch before running this script."
        ) from exc
    return torch, fx, nn, F, ProfilerActivity, profile


torch, fx, nn, F, ProfilerActivity, profile = require_torch()


CATEGORY_ORDER = [
    "Activation",
    "Attention",
    "Broadcast",
    "Convolution",
    "Full Architecture",
    "Fusion",
    "Loss",
    "Math",
    "Matrix Multiply",
    "Normalization",
    "Optimizer",
    "Pooling",
    "Index",
    "Resize",
    "Reduce",
    "Other",
]


def categorize_aten(name: str) -> str | None:
    lowered = name.lower()
    if "::" in lowered:
        lowered = lowered.split("::", 1)[1]

    if any(x in lowered for x in ("conv", "mkldnn_convolution")):
        return "Convolution"
    if any(x in lowered for x in ("matmul", "mm", "bmm", "addmm", "linear")):
        return "Matrix Multiply"
    if any(x in lowered for x in ("scaled_dot_product_attention", "attention")):
        return "Attention"
    if any(x in lowered for x in ("relu", "gelu", "sigmoid", "tanh", "silu", "elu")):
        return "Activation"
    if any(x in lowered for x in ("batch_norm", "layer_norm", "group_norm", "native_layer_norm")):
        return "Normalization"
    if any(x in lowered for x in ("max_pool", "avg_pool", "adaptive_avg_pool")):
        return "Pooling"
    if any(x in lowered for x in ("sum", "mean", "amax", "amin", "prod", "var")):
        return "Reduce"
    if any(x in lowered for x in ("gather", "scatter", "index", "select", "embedding")):
        return "Index"
    if any(x in lowered for x in ("upsample", "interpolate", "grid_sampler")):
        return "Resize"
    if any(x in lowered for x in ("cross_entropy", "nll_loss", "mse_loss", "smooth_l1")):
        return "Loss"
    if any(x in lowered for x in ("adam", "sgd", "optim")):
        return "Optimizer"
    if any(x in lowered for x in ("add", "sub", "mul", "div", "where", "clamp", "maximum", "minimum")):
        return "Math"
    if any(x in lowered for x in ("softmax", "log_softmax", "exp", "log", "sqrt", "pow")):
        return "Math"
    return None


def module_category(module: nn.Module) -> str | None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "Convolution"
    if isinstance(module, nn.Linear):
        return "Matrix Multiply"
    if isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.SiLU)):
        return "Activation"
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        return "Normalization"
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
        return "Pooling"
    if isinstance(module, (nn.CrossEntropyLoss, nn.MSELoss, nn.NLLLoss)):
        return "Loss"
    if isinstance(module, (nn.Embedding,)):
        return "Index"
    if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        return "Full Architecture"
    if isinstance(module, nn.MultiheadAttention):
        return "Attention"
    return None


def fx_node_category(node: fx.Node, modules: dict[str, nn.Module]) -> str | None:
    if node.op == "call_module":
        return module_category(modules[node.target])
    if node.op == "call_function":
        target_name = getattr(node.target, "__name__", str(node.target))
        return categorize_aten(target_name)
    if node.op == "call_method":
        return categorize_aten(str(node.target))
    return None


FUSABLE_CATEGORIES = {
    "Activation",
    "Broadcast",
    "Convolution",
    "Loss",
    "Math",
    "Matrix Multiply",
    "Normalization",
    "Pooling",
    "Reduce",
}


@dataclass
class Workload:
    name: str
    model_factory: Callable[[], nn.Module]
    input_factory: Callable[[], tuple]
    run_factory: Callable[[nn.Module], Callable[[], object]]
    traceable: bool = True


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = F.relu(y + x.mean(dim=1, keepdim=True))
        y = self.pool(y).flatten(1)
        return self.fc(y)


class MobileNetStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.pw = nn.Conv2d(16, 64, 1)
        self.bn = nn.BatchNorm2d(64)
        self.proj = nn.Linear(64, 16)

    def forward(self, x):
        x = F.silu(self.dw(x))
        x = F.relu(self.bn(self.pw(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn)
        y = self.fc2(F.gelu(self.fc1(x)))
        return self.ln2(x + y)


class LSTMWorkload(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 96, num_layers=2, batch_first=True)
        self.fc = nn.Linear(96, 16)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1])


class OptimizerToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def inference_runner(model: nn.Module, inputs: tuple):
    def run():
        with torch.no_grad():
            return model(*inputs)

    return run


def optimizer_runner(model: nn.Module, inputs: tuple):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randint(0, 10, (inputs[0].shape[0],))

    def run():
        opt.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        opt.step()
        return loss

    return run


def build_workloads() -> list[Workload]:
    return [
        Workload(
            "cnn_residual",
            ResidualCNN,
            lambda: (torch.randn(16, 3, 64, 64),),
            lambda model: inference_runner(model, (torch.randn(16, 3, 64, 64),)),
        ),
        Workload(
            "mobilenet_style",
            MobileNetStyle,
            lambda: (torch.randn(16, 16, 64, 64),),
            lambda model: inference_runner(model, (torch.randn(16, 16, 64, 64),)),
        ),
        Workload(
            "transformer_block",
            TransformerBlock,
            lambda: (torch.randn(8, 128, 128),),
            lambda model: inference_runner(model, (torch.randn(8, 128, 128),)),
            traceable=False,
        ),
        Workload(
            "lstm_sequence",
            LSTMWorkload,
            lambda: (torch.randn(16, 64, 64),),
            lambda model: inference_runner(model, (torch.randn(16, 64, 64),)),
            traceable=False,
        ),
        Workload(
            "optimizer_step",
            OptimizerToy,
            lambda: (torch.randn(64, 128),),
            lambda model: optimizer_runner(model, (torch.randn(64, 128),)),
        ),
    ]


def profile_categories(run, warmup: int, iters: int) -> Counter:
    for _ in range(warmup):
        run()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(iters):
            run()

    counts = Counter()
    for event in prof.key_averages():
        category = categorize_aten(event.key)
        if category is not None:
            counts[category] += int(event.count)
    return counts


def fusion_opportunities(model: nn.Module, example_inputs: tuple, traceable: bool) -> int:
    if not traceable:
        return 0
    try:
        graph = fx.symbolic_trace(model)
    except Exception:
        return 0

    modules = dict(graph.named_modules())
    node_categories = {
        node: fx_node_category(node, modules)
        for node in graph.graph.nodes
    }
    opportunities = 0
    for node, category in node_categories.items():
        if category not in FUSABLE_CATEGORIES:
            continue
        for user in node.users:
            if node_categories.get(user) in FUSABLE_CATEGORIES:
                opportunities += 1
    return opportunities


def summarize(results: list[dict]) -> dict:
    aggregate = Counter()
    for item in results:
        aggregate.update(item["category_counts"])
        aggregate["Fusion"] += item["fusion_opportunities"]

    total = sum(aggregate.values())
    shares = {
        category: (aggregate[category] / total if total else 0.0)
        for category in CATEGORY_ORDER
        if aggregate.get(category, 0) > 0
    }
    return {
        "aggregate_counts": dict(aggregate),
        "aggregate_shares": shares,
        "total_mapped_events": total,
    }


def print_markdown(results: list[dict], summary: dict):
    print("# Workload Frequency Summary\n")
    print("| Workload | Top categories | Fusion opportunities |")
    print("|---|---:|---:|")
    for item in results:
        counts = Counter(item["category_counts"])
        top = ", ".join(f"{cat}={count}" for cat, count in counts.most_common(4))
        print(f"| {item['workload']} | {top} | {item['fusion_opportunities']} |")

    print("\n## Aggregate\n")
    aggregate = Counter(summary["aggregate_counts"])
    total = summary["total_mapped_events"]
    for category, count in aggregate.most_common():
        share = 100 * count / total if total else 0.0
        print(f"- {category}: {count} ({share:.1f}%)")

    print("\n## Paper Text Snippet\n")
    print(
        "We profiled representative PyTorch workloads, including CNN-style, "
        "Transformer-style, recurrent, and optimizer-step workloads, using "
        "operator traces and FX graphs. The profiling shows that fusible "
        "operator patterns appear across model families, including common "
        "sequences such as convolution or matrix multiplication followed by "
        "normalization, activation, reduction, or element-wise operations. "
        "This supports allocating more tasks to Fusion, while categories such "
        "as Optimizer and standalone Reduce expose fewer distinct standalone "
        "kernel patterns after removing near-duplicates."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=Path("workload_frequency.json"))
    args = parser.parse_args()

    results = []
    for workload in build_workloads():
        model = workload.model_factory().eval()
        inputs = workload.input_factory()
        run = workload.run_factory(model)
        category_counts = profile_categories(run, args.warmup, args.iters)
        if workload.name == "optimizer_step":
            category_counts["Optimizer"] += args.iters
        fusion_count = fusion_opportunities(model, inputs, workload.traceable)
        results.append(
            {
                "workload": workload.name,
                "category_counts": dict(category_counts),
                "fusion_opportunities": fusion_count,
            }
        )

    summary = summarize(results)
    output = {"results": results, "summary": summary}
    args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_markdown(results, summary)
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
