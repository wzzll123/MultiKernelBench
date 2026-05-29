import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that selects elements from a tensor along a dimension using indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        """
        Selects elements from input tensor along the specified dimension using index.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension in which to index.
            index (torch.Tensor): The 1-D tensor containing the indices to index.

        Returns:
            torch.Tensor: Tensor with selected elements.
        """
        return torch.index_select(x, dim, index)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "18_index.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        index_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        dim = dim_info["value"]
        
        index_range = index_info.get("range", [0, x_info["shape"][dim] - 1])
        index_size = index_info["shape"][0]
        index = torch.randint(index_range[0], index_range[1] + 1, (index_size,), dtype=torch.int64)
        
        input_groups.append([x, dim, index])
    return input_groups


def get_init_inputs():
    return []
