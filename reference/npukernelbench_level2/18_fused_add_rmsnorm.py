import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Add + RMSNorm operation. Computes residual addition followed by Root Mean Square Normalization.
    Formula: x = hidden_states + residual; y = (x * rsqrt(mean(x^2) + eps)) * weight
    Used in models like Qwen3-30B-A3B, Llama-3.1-8B, DeepSeek-V3/R1.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Applies fused add + RMSNorm to the input tensors.

        Args:
            hidden_states (torch.Tensor): Input tensor with shape [batch_size, hidden_size].
                                          Supports bfloat16.
            residual (torch.Tensor): Residual tensor with shape [batch_size, hidden_size].
                                     Supports bfloat16.
            weight (torch.Tensor): Weight tensor with shape [hidden_size].
                                   Supports bfloat16.
            eps (float, optional): Value added to denominator for numerical stability. Default: 1e-6.

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, hidden_size], same dtype as hidden_states.
        """
        x = hidden_states.to(torch.float32) + residual.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x * inv_rms) * weight.to(torch.float32)
        return y.to(hidden_states.dtype)


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
