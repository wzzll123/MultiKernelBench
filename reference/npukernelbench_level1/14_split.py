import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that splits a tensor into chunks.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, split_size_or_sections, dim: int = 0):
        """
        Splits the tensor into chunks.

        Args:
            x (torch.Tensor): Input tensor to split.
            split_size_or_sections (int or list): If int, size of each chunk. If list, sizes of each chunk.
            dim (int, optional): Dimension along which to split the tensor.

        Returns:
            tuple: Tuple of tensors resulting from the split.
        """
        return torch.split(x, split_size_or_sections, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "14_split.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        split_info = inputs[1]
        dim_info = inputs[2]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        split_size_or_sections = split_info["value"]
        dim = dim_info["value"]
        input_groups.append([x, split_size_or_sections, dim])
    return input_groups


def get_init_inputs():
    return []
