import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model that computes ReLU backward (gradient).
    torch.relu_backward(grad_output, self) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, self_: torch.Tensor) -> torch.Tensor:
        """
        Computes ReLU backward.

        Args:
            grad_output (torch.Tensor): Gradient from upstream.
            self_ (torch.Tensor): Original input to ReLU forward.

        Returns:
            torch.Tensor: Gradient w.r.t. ReLU input.
        """
        return torch.ops.aten.relu_backward(grad_output, self_)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "20_relu_backward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    for case in cases:
        inputs = case["inputs"]
        grad_info = inputs[0]
        self_info = inputs[1]

        grad_dtype = dtype_map[grad_info["dtype"]]
        self_dtype = dtype_map[self_info["dtype"]]
        grad_output = torch.randn(grad_info["shape"], dtype=grad_dtype)
        self_ = torch.randn(self_info["shape"], dtype=self_dtype)
        input_groups.append([grad_output, self_])
    return input_groups


def get_init_inputs():
    return []
