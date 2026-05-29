import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that gathers values along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, sparse_grad: bool = False) -> torch.Tensor:
        """
        Gathers values along an axis specified by dim.

        Args:
            x (torch.Tensor): Input tensor (src).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to gather.
            sparse_grad (bool, optional): If True, gradient w.r.t. input will be a sparse tensor.

        Returns:
            torch.Tensor: Tensor with gathered values, same shape as index.
        """
        return torch.gather(x, dim, index, sparse_grad=sparse_grad)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "20_gather.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        index_info = inputs[2]
        sparse_grad_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        dim = dim_info["value"]
        
        index_range = index_info.get("range", [0, x_info["shape"][dim] - 1])
        index = torch.randint(index_range[0], index_range[1] + 1, tuple(index_info["shape"]), dtype=torch.int64)
        sparse_grad = sparse_grad_info["value"]
        
        input_groups.append([x, dim, index, sparse_grad])
    return input_groups


def get_init_inputs():
    return []
