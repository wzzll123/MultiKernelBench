import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Swish activation function.
    x / (1 + exp(-beta * x)) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            beta (float, optional): Scale factor. Default: 1.0.

        Returns:
            torch.Tensor: Output tensor after Swish activation.
        """
        return x / (1 + torch.exp(-beta * x))


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "12_swish.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        beta_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        beta = float(beta_info["value"])
        input_groups.append([x, beta])
    return input_groups


def get_init_inputs():
    return []
