import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the negative log likelihood loss.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor = None, ignore_index: int = -100,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the negative log likelihood loss.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C) or (N, C, d1, d2, ...).
            target (torch.Tensor): Target tensor of shape (N,) or (N, d1, d2, ...).
            weight (torch.Tensor, optional): Manual rescaling weight given to each class.
            ignore_index (int, optional): Target value that is ignored and does not contribute to gradient.
            reduction (str, optional): Reduction method ('none', 'mean', 'sum').

        Returns:
            torch.Tensor: NLL loss value.
        """
        return torch.nn.functional.nll_loss(input, target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "25_nll_loss.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        input_info = inputs[0]
        target_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[input_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            input_tensor = torch.normal(mu, sigma, input_info["shape"], dtype=dtype) + torch.ones(input_info["shape"], dtype=dtype)
        else:
            input_tensor = torch.empty(input_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(input_info["shape"], dtype=dtype)
        
        target_range = target_info.get("range", [0, input_info["shape"][1] - 1])
        target = torch.randint(target_range[0], target_range[1] + 1, tuple(target_info["shape"]), dtype=torch.int64)
        
        weight = None
        ignore_index = -100
        reduction = "mean"
        
        for inp in inputs[2:]:
            if inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "ignore_index":
                ignore_index = inp["value"]
            elif inp["name"] == "reduction":
                reduction = inp["value"]
        
        input_groups.append([input_tensor, target, weight, ignore_index, reduction])
    return input_groups


def get_init_inputs():
    return []
