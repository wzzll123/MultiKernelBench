import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Applies the Threshold activation function.
    torch.threshold(input, threshold, value) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, threshold, value) -> torch.Tensor:
        """
        Applies threshold activation.

        Args:
            x (torch.Tensor): Input tensor.
            threshold: Threshold value.
            value: Replacement value.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.threshold(x, threshold, value)


def _get_scalar(info):
    val = info.get("value")
    if val is not None:
        return val
    rv = info.get("range_values")
    if isinstance(rv, (int, float)):
        return rv
    if isinstance(rv, list) and len(rv) > 0:
        return rv[0]
    dtype_str = info.get("dtype", "")
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        return 1
    return 1.0


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "19_threshold.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        th_info = inputs[1]
        val_info = inputs[2]

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp64": torch.float64,
            "float64": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
        }

        x_dtype_str = x_info["dtype"]
        x_dtype = dtype_map[x_dtype_str]
        x_shape = x_info["shape"]

        if x_dtype_str.startswith("uint"):
            x = torch.randint(0, 10, x_shape, dtype=torch.uint8)
        elif x_dtype_str.startswith("int"):
            x = torch.randint(-10, 10, x_shape, dtype=x_dtype)
        else:
            x = torch.randn(x_shape, dtype=x_dtype)

        th_val = _get_scalar(th_info)
        val_val = _get_scalar(val_info)

        # Cast scalars to appropriate Python types based on dtype
        if x_dtype_str.startswith("int") or x_dtype_str.startswith("uint"):
            th_val = int(th_val)
            val_val = int(val_val)
        else:
            th_val = float(th_val)
            val_val = float(val_val)

        input_groups.append([x, th_val, val_val])
    return input_groups


def get_init_inputs():
    return []
