import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that performs Rotary Position Embedding multiplication.
    torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode='half') -> Tensor

    Pure PyTorch implementation (replacing torch_npu.npu_moe_compute_expert_tokens):
    if rotary_mode == 'half':
            half_d = input.shape[-1] // 2
            x1 = input[..., :half_d]
            x2 = input[..., half_d:]
            rotated = torch.cat((-x2, x1), dim=-1)
        elif rotary_mode == 'interleave':
            x1 = input[..., 0::2]
            x2 = input[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return input * r1 + rotated * r2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = 'half') -> torch.Tensor:
        """
        Applies rotary position embedding multiplication to the input tensor.

        Args:
            input (torch.Tensor): Input tensor, must be 4D. Supports float16, bfloat16, float32.
            r1 (torch.Tensor): Cosine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            r2 (torch.Tensor): Sine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            rotary_mode (str, optional): Computation mode, supports 'half' and 'interleave'. Default: 'half'.

        Returns:
            torch.Tensor: Output tensor with rotary position embedding applied.
        """
        return torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode=rotary_mode)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_rotary_mul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        r1_info = inputs[1]
        r2_info = inputs[2]
        rotary_mode_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[input_info["dtype"]]
        
        inp = torch.randn(input_info["shape"], dtype=dtype)
        r1 = torch.randn(r1_info["shape"], dtype=dtype)
        r2 = torch.randn(r2_info["shape"], dtype=dtype)
        rotary_mode = rotary_mode_info["value"]
        input_groups.append([inp, r1, r2, rotary_mode])
    return input_groups


def get_init_inputs():
    return []
