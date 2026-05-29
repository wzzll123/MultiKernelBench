import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats a tensor along specified dimensions.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats: tuple) -> torch.Tensor:
        """
        Repeats the tensor along each dimension.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (tuple): Number of repeats for each dimension.

        Returns:
            torch.Tensor: Repeated tensor.
        """
        return x.repeat(*repeats)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "16_repeat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        repeats_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        repeats = tuple(repeats_info["value"])
        input_groups.append([x, repeats])
    return input_groups


def get_init_inputs():
    return []
