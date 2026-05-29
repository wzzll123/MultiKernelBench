import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that concatenates tensors along a dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, dim: int = 0) -> torch.Tensor:
        """
        Concatenates the given sequence of tensors in the given dimension.

        Args:
            tensors (list): List of tensors to concatenate. All tensors must have the same shape except in the concatenating dimension.
            dim (int, optional): The dimension over which the tensors are concatenated.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.cat(tensors, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_cat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        tensors_info = inputs[0]
        dim_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[tensors_info["dtype"]]
        
        shapes = tensors_info["shapes"]
        tensors = [torch.randn(shape, dtype=dtype) for shape in shapes]
        dim = dim_info["value"]
        input_groups.append([tensors, dim])
    return input_groups


def get_init_inputs():
    return []
