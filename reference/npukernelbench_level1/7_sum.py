# torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.sum.html#torch.sum

import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the sum of elements along specified dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        """
        Returns the sum of elements along specified dimensions.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int or tuple of ints, optional): Dimension(s) to reduce.
            keepdim (bool): Whether to keep the reduced dimension(s).

        Returns:
            torch.Tensor: Tensor with sum along specified dimensions.
        """
        return torch.sum(x, dim=dim, keepdim=keepdim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_sum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        keepdim_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
        dim = dim_info["value"]
        keepdim = keepdim_info["value"]
        input_groups.append([x, dim, keepdim])
    return input_groups


def get_init_inputs():
    return []
