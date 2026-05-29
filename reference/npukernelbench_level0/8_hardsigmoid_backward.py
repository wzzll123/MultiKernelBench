import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Hardsigmoid backward.
    torch.ops.aten.hardsigmoid_backward(grad_output, self) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, self_x: torch.Tensor) -> torch.Tensor:
        """
        Applies Hardsigmoid backward.

        Args:
            grad_output (torch.Tensor): Gradient from upstream.
            self_x (torch.Tensor): Input tensor from forward.

        Returns:
            torch.Tensor: Gradient w.r.t. input.
        """
        return torch.ops.aten.hardsigmoid_backward(grad_output, self_x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_hardsigmoid_backward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        g_info = inputs[0]
        x_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        g_dtype = dtype_map[g_info["dtype"]]
        x_dtype = dtype_map[x_info["dtype"]]

        g = torch.randn(g_info["shape"], dtype=g_dtype)
        x = torch.randn(x_info["shape"], dtype=x_dtype)
        input_groups.append([g, x])
    return input_groups


def get_init_inputs():
    return []
