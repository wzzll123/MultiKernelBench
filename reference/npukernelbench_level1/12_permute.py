import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs a permutation of tensor dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dims: tuple) -> torch.Tensor:
        """
        Permutes the dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with at least the number of dimensions in dims.
            dims (tuple): The desired ordering of dimensions.

        Returns:
            torch.Tensor: Tensor with permuted dimensions.
        """
        return torch.permute(x, dims).contiguous()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "12_permute.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dims_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        dims = tuple(dims_info["value"])
        input_groups.append([x, dims])
    return input_groups


def get_init_inputs():
    return []
