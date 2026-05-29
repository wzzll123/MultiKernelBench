# torch.histc(input, bins=100, min=0, max=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.histc.html#torch.histc

import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the histogram of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int, min: int, max: int) -> torch.Tensor:
        """
        Computes the histogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            bins (int): Number of histogram bins.
            min (int): Lower end of the range (inclusive).
            max (int): Upper end of the range (inclusive).

        Returns:
            torch.Tensor: Histogram tensor of shape (bins,).
        """
        return torch.histc(x, bins=bins, min=min, max=max)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_histc.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        bins_info = inputs[1]
        min_info = inputs[2]
        max_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        bins = bins_info["value"]
        min_val = min_info["value"]
        max_val = max_info["value"]
        input_groups.append([x, bins, min_val, max_val])
    return input_groups


def get_init_inputs():
    return []
