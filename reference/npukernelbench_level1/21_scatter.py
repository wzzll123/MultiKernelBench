import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that scatters values from src into a tensor at specified indices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, reduce: str = None) -> torch.Tensor:
        """
        Scatters values from src into x at the indices specified in index.

        Args:
            x (torch.Tensor): Input tensor (self, will be cloned to avoid in-place modification).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to scatter.
            src (torch.Tensor): The source elements to scatter.
            reduce (str, optional): Reduction operation ('add', 'multiply').

        Returns:
            torch.Tensor: Tensor with scattered values.
        """
        if reduce is not None:
            x.scatter_(dim, index, src, reduce=reduce)
        else:
            x.scatter_(dim, index, src)
        return x


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "21_scatter.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        index_info = inputs[2]
        src_info = inputs[3]
        reduce_info = inputs[4]
        
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
        
        src = torch.randn(src_info["shape"], dtype=dtype)
        reduce = reduce_info["value"]
        
        input_groups.append([x, dim, index, src, reduce])
    return input_groups


def get_init_inputs():
    return []
