import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Applies the Hardswish activation function.
    F.hardswish(input) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Hardswish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Hardswish activation.
        """
        return F.hardswish(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "16_hardswish.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        dtype_str = x_info["dtype"]
        dtype = dtype_map[dtype_str]
        shape = x_info["shape"]
        x = torch.randn(shape, dtype=dtype)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
