import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs batch matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies batch matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Input tensor of shape (batch, m, n).
            B (torch.Tensor): Input tensor of shape (batch, n, p).

        Returns:
            torch.Tensor: Output tensor of shape (batch, m, p) after performing torch.bmm.
        """
        return torch.bmm(A, B)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_batch_matmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        A_info = inputs[0]
        B_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[A_info["dtype"]]
        
        A = torch.randn(A_info["shape"], dtype=dtype)
        B = torch.randn(B_info["shape"], dtype=dtype)
        input_groups.append([A, B])
    return input_groups


def get_init_inputs():
    return []
