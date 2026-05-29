import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Layer Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, normalized_shape: list, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [*, normalized_shape[0], ...].
            normalized_shape (list): Shape over which to normalize.
            weight (torch.Tensor, optional): Weight tensor of shape normalized_shape.
            bias (torch.Tensor, optional): Bias tensor of shape normalized_shape.

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "10_layer_norm.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        
        normalized_shape = None
        weight = None
        bias = None
        
        for inp in inputs[1:]:
            if inp["name"] == "normalized_shape":
                normalized_shape = inp["value"]
            elif inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "bias":
                bias = torch.randn(inp["shape"], dtype=dtype)
        
        input_groups.append([x, normalized_shape, weight, bias])
    return input_groups


def get_init_inputs():
    return []
