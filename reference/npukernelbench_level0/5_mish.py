import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Mish activation function.
    torch.nn.functional.mish(input) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Mish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Mish activation.
        """
        return torch.nn.functional.mish(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_mish.json")
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
            "float64": torch.float64,
        }
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
