import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Batched 2D RoPE Position Encoding Backward Module.

    Computes the backward pass (gradient) for batched 2D Rotary Position Encoding.
    Uses chain rule to compute gradient w.r.t. idx_theta:
    - d(cos(x))/dx = -sin(x)
    - d(sin(x))/dx = cos(x)
    Therefore: grad_idx_theta = -grad_cos * sin(idx_theta) + grad_sin * cos(idx_theta)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_cos: torch.Tensor, grad_sin: torch.Tensor, idx_theta: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for batched 2D RoPE position encoding.

        Args:
            grad_cos: Gradient w.r.t. cos output [batch_size, seq_len, head_dim]
            grad_sin: Gradient w.r.t. sin output [batch_size, seq_len, head_dim]
            idx_theta: Saved angles from forward pass [batch_size, seq_len, head_dim]

        Returns:
            grad_idx_theta: Gradient w.r.t. idx_theta [batch_size, seq_len, head_dim]
        """
        sin_theta = torch.sin(idx_theta)
        cos_theta = torch.cos(idx_theta)
        grad_cos_f32 = grad_cos.to(torch.float32)
        grad_sin_f32 = grad_sin.to(torch.float32)
        grad_idx_theta = -grad_cos_f32 * sin_theta + grad_sin_f32 * cos_theta
        return grad_idx_theta


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
