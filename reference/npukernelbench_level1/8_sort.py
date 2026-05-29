# torch.sort(input, dim=-1, descending=False, *, stable=False, out=None)
# https://docs.pytorch.org/docs/stable/generated/torch.sort.html#torch.sort

import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that sorts a tensor along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
        """
        Sorts the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int, optional): The dimension to sort along.
            descending (bool, optional): Controls the sorting order (ascending by default).

        Returns:
            torch.Tensor: Sorted tensor with same shape as input.
        """
        return torch.sort(x, dim=dim, descending=descending)[0]


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_sort.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        descending_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        dim = dim_info["value"]
        descending = descending_info["value"]
        input_groups.append([x, dim, descending])
    return input_groups


def get_init_inputs():
    return []
