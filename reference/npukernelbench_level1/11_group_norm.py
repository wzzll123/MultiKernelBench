import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Group Normalization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, num_groups: int, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Group Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, *].
            num_groups (int): Number of groups to separate channels into.
            weight (torch.Tensor, optional): Weight tensor of shape [C].
            bias (torch.Tensor, optional): Bias tensor of shape [C].

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "11_group_norm.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(x_info["shape"], dtype=dtype)
        
        num_groups = None
        weight = None
        bias = None
        
        for inp in inputs[1:]:
            if inp["name"] == "num_groups":
                num_groups = inp["value"]
            elif inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "bias":
                bias = torch.randn(inp["shape"], dtype=dtype)
        
        input_groups.append([x, num_groups, weight, bias])
    return input_groups


def get_init_inputs():
    return []
