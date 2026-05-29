import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Element-wise not-equal comparison with a scalar.
    torch.ne(input, other) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, other) -> torch.Tensor:
        """
        Applies element-wise not-equal comparison.

        Args:
            x (torch.Tensor): Input tensor.
            other: Scalar value to compare against.

        Returns:
            torch.Tensor: Boolean tensor result of x != other.
        """
        return torch.ne(x, other)


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
    if dtype_str == "bool":
        return True
    if dtype_str.startswith("int") or dtype_str.startswith("uint"):
        return 1
    return 1.0


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "18_ne_scalar.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        other_info = inputs[1]

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
            "uint16": torch.uint16,
            "uint32": torch.uint32,
            "uint64": torch.uint64,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }

        x_dtype_str = x_info["dtype"]
        x_dtype = dtype_map[x_dtype_str]
        x_shape = x_info["shape"]

        if x_dtype == torch.bool:
            x = torch.randint(0, 2, x_shape, dtype=torch.bool)
        elif x_dtype_str.startswith("uint"):
            x = torch.randint(0, 10, x_shape, dtype=torch.int64).to(x_dtype)
        elif x_dtype_str.startswith("int"):
            x = torch.randint(-10, 10, x_shape, dtype=x_dtype)
        elif x_dtype_str.startswith("complex"):
            x = torch.randn(x_shape, dtype=x_dtype)
        else:
            x = torch.randn(x_shape, dtype=x_dtype)

        other_val = _get_scalar(other_info)
        other_dtype_str = other_info.get("dtype", "")
        if other_dtype_str == "bool":
            other_val = bool(other_val)
        elif other_dtype_str.startswith("complex"):
            other_val = complex(other_val)
        elif other_dtype_str.startswith("int") or other_dtype_str.startswith("uint"):
            other_val = int(other_val)
        else:
            other_val = float(other_val)

        input_groups.append([x, other_val])
    return input_groups


def get_init_inputs():
    return []
