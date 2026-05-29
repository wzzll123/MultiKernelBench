import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that returns indices of non-zero elements.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, as_tuple: bool = False):
        """
        Returns indices of non-zero elements in the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            as_tuple (bool, optional): If True, returns a tuple of 1-D tensors, one for each dimension.

        Returns:
            torch.Tensor or tuple: Indices of non-zero elements.
        """
        return torch.nonzero(x, as_tuple=as_tuple)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "22_nonzero.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        as_tuple_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        dtype = dtype_map.get(x_info["dtype"], torch.float32)
        
        if dtype in [torch.int32, torch.int64]:
            x = torch.randint(-10, 10, x_info["shape"], dtype=dtype)
        else:
            x = torch.randn(x_info["shape"], dtype=dtype)
        as_tuple = as_tuple_info["value"]
        input_groups.append([x, as_tuple])
    return input_groups


def get_init_inputs():
    return []
