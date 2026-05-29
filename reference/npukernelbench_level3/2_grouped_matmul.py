import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model that performs grouped matrix multiplication using torch.nn.functional.grouped_mm.
    Supports both 3D inputs (direct grouping) and 2D inputs with offset-based grouping.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor = None) -> torch.Tensor:
        """
        Applies grouped matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Left operand tensor.
                - 3D shape: (num_groups, m, k) - groups are directly enumerated
                - 2D shape: (total_rows, k) - rows are grouped according to offsets
            B (torch.Tensor): Right operand tensor.
                - Shape: (num_groups, k, n) for common forward pass (out = input @ weight.T)
            offsets (torch.Tensor): 1D int32 tensor defining group start indices in A.
                - Required when A is 2D (total_rows, k).
                - Length: exactly num_groups (matches first dimension of B).
                - Format: [start_0, start_1, start_2, ..., start_{G-1}]
                - Group i uses rows A[offsets[i] : offsets[i+1]] for i < G-1.
                - Last group uses rows A[offsets[-1] : total_rows].
                - Must be strictly increasing, offsets[0] = 0.

        Returns:
            torch.Tensor: Concatenated results of each per-group GEMM operation.
        """
        if hasattr(F, 'grouped_mm'):
            return F.grouped_mm(A, B, offs=offsets)
        else:
            return torch._grouped_mm(A, B, offs=offsets)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_grouped_matmul.json")
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
        
        if len(inputs) > 2:
            offsets_info = inputs[2]
            if offsets_info.get("type") == "tensor":
                total_rows = A_info["shape"][0]
                num_groups = B_info["shape"][0]
                group_size = total_rows // num_groups
                offsets = torch.tensor([i * group_size for i in range(num_groups)], dtype=torch.int32)
                input_groups.append([A, B, offsets])
            else:
                input_groups.append([A, B, None])
        else:
            input_groups.append([A, B, None])
    return input_groups


def get_init_inputs():
    return []