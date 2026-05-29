import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs element-wise logical NOT.
    torch.logical_not(input) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise logical NOT to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output bool tensor of the logical NOT result.
        """
        return torch.logical_not(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_logical_not.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bool": torch.bool,
            "int32": torch.int32,
            "int8": torch.int8,
            "int16": torch.int16,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "float64": torch.float64,
        }
        dtype = dtype_map[x_info["dtype"]]

        if dtype == torch.bool:
            x = torch.randint(0, 2, x_info["shape"], dtype=torch.bool)
        elif dtype == torch.uint8:
            x = torch.randint(0, 256, x_info["shape"], dtype=dtype)
        elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.randint(-10, 11, x_info["shape"], dtype=dtype)
        else:
            x = torch.randn(x_info["shape"], dtype=dtype)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
