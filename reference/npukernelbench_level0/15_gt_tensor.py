import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Element-wise greater-than comparison.
    torch.gt(input, other) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise greater-than comparison.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Boolean tensor result of x > y.
        """
        # uint16/32/64 are not supported by torch.gt on CPU; cast to float32 as the reference executor does
        if x.dtype in (torch.uint16, torch.uint32, torch.uint64) or y.dtype in (torch.uint16, torch.uint32, torch.uint64):
            return torch.gt(x.to(torch.float32), y.to(torch.float32))
        return torch.gt(x, y)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_gt_tensor.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]

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
        }

        x_dtype_str = x_info["dtype"]
        y_dtype_str = y_info["dtype"]
        x_dtype = dtype_map[x_dtype_str]
        y_dtype = dtype_map[y_dtype_str]
        x_shape = x_info["shape"]
        y_shape = y_info["shape"]

        if x_dtype == torch.bool:
            x = torch.randint(0, 2, x_shape, dtype=torch.bool)
        elif x_dtype_str.startswith("uint"):
            x = torch.randint(0, 10, x_shape, dtype=torch.int64).to(x_dtype)
        elif x_dtype_str.startswith("int"):
            x = torch.randint(-10, 10, x_shape, dtype=x_dtype)
        else:
            x = torch.randn(x_shape, dtype=x_dtype)

        if y_dtype == torch.bool:
            y = torch.randint(0, 2, y_shape, dtype=torch.bool)
        elif y_dtype_str.startswith("uint"):
            y = torch.randint(0, 10, y_shape, dtype=torch.int64).to(y_dtype)
        elif y_dtype_str.startswith("int"):
            y = torch.randint(-10, 10, y_shape, dtype=y_dtype)
        else:
            y = torch.randn(y_shape, dtype=y_dtype)

        input_groups.append([x, y])
    return input_groups


def get_init_inputs():
    return []
