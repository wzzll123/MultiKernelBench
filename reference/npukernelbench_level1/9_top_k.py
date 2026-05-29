import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that finds the k largest elements along a given dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> torch.Tensor:
        """
        Finds the k largest/smallest elements along a given dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            k (int): The number of elements to return.
            dim (int, optional): The dimension to find the top k along.
            largest (bool, optional): If True, return the largest elements.
            sorted (bool, optional): If True, the elements are returned sorted.

        Returns:
            torch.Tensor: The top k values tensor.
        """
        return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted)[0]


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_top_k.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        k_info = inputs[1]
        dim_info = inputs[2]
        largest_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(x_info["shape"], dtype=dtype)
        k = k_info["value"]
        dim = dim_info["value"]
        largest = largest_info["value"]
        input_groups.append([x, k, dim, largest])
    return input_groups


def get_init_inputs():
    return []
