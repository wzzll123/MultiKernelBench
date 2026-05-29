import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs element-wise addition with broadcasting support.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        """
        return torch.add(x, y, alpha=alpha)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_add.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]
        alpha_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        y = torch.randn(y_info["shape"], dtype=dtype)
        alpha = alpha_info["value"]
        input_groups.append([x, y, alpha])
    return input_groups


def get_init_inputs():
    return []
