import torch
import torch.nn as nn
import json
import os
import numpy as np

class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        NPU torch.cumsum may use a parallel scan that differs from the serial
        AscendC kernel for float32/float16. We compute the reference on CPU with
        serial accumulation (matching the kernel) to ensure a fair comparison.
        bfloat16 still uses NPU torch.cumsum.
        """
        if x.dtype == torch.float32:
            out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float32)
            return torch.from_numpy(out).to(x.device)
        if x.dtype == torch.float16:
            out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float16)
            return torch.from_numpy(out).to(x.device)
        return torch.cumsum(x, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_cumsum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.distributions.Uniform(0.0, 1.0).sample(x_info["shape"]).to(dtype)
        dim = dim_info["value"]
        input_groups.append([x, dim])
    return input_groups


def get_init_inputs():
    return []
