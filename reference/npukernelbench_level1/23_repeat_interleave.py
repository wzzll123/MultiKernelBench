import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats elements of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats, dim: int = None, output_size: int = None) -> torch.Tensor:
        """
        Repeats elements of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (int or torch.Tensor): Number of repetitions for each element.
            dim (int, optional): The dimension along which to repeat values.
            output_size (int, optional): Total output size for the repeated dimension.

        Returns:
            torch.Tensor: Tensor with repeated elements.
        """
        if output_size is not None:
            return torch.repeat_interleave(x, repeats, dim=dim, output_size=output_size)
        return torch.repeat_interleave(x, repeats, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "23_repeat_interleave.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        repeats_info = inputs[1]
        dim_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        repeats = repeats_info["value"]
        dim = dim_info["value"]
        input_groups.append([x, repeats, dim])
    return input_groups


def get_init_inputs():
    return []
