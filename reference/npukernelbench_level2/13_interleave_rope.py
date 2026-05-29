import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs interleave RoPE (Rotary Position Embedding).
    torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
    PyTorch native implementation of forward function
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f32 = x.float()
        cos_f32 = cos.float()
        sin_f32 = sin.float()

        B, N, S, D = x_f32.shape

        x_reshaped = x_f32.reshape(B, N, S, D // 2, 2)
        x_transposed = x_reshaped.transpose(-1, -2)
        x_interleaved = x_transposed.reshape(B, N, S, D)

        cos_expanded = cos_f32
        sin_expanded = sin_f32

        if cos_expanded.shape[2] == 1 and S > 1:
            cos_expanded = cos_expanded.expand(B, N, S, D)
        if sin_expanded.shape[2] == 1 and S > 1:
            sin_expanded = sin_expanded.expand(B, N, S, D)

        x_rotated = torch.zeros_like(x_interleaved)
        x_rotated[..., :D // 2] = -x_interleaved[..., D // 2:]
        x_rotated[..., D // 2:] = x_interleaved[..., :D // 2]

        output_f32 = x_interleaved * cos_expanded + x_rotated * sin_expanded

        return output_f32.to(orig_dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Performs interleave RoPE on input tensor.

        Args:
            x (torch.Tensor): Input tensor to process. Must be 4D with shape (B, N, S, D).
                              dtype: bfloat16, float16, format: ND.
                              Does not support non-contiguous tensors.
            cos (torch.Tensor): RoPE cosine component. Must be 4D with shape (B, N, S, D).
                                S can be 1 or same as x's S. dtype and format must match x.
                                Does not support non-contiguous tensors.
            sin (torch.Tensor): RoPE sine component. Shape, dtype and format must match cos.
                                Does not support non-contiguous tensors.

        Returns:
            torch.Tensor: Output tensor after interleave RoPE, same shape as input x.
        """
        return torch_npu.npu_interleave_rope(x, cos, sin)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
