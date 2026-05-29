import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Masked Softmax with Attention Dropout Backward Module.

    Backward pass for masked softmax with dropout.
    Computes gradients through three operations in reverse order:
    1. Dropout backward: scale by dropout mask and inverse probability
    2. Softmax backward: y * (grad - sum(y * grad))
    3. Masked fill backward: zero out gradients at masked positions
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        p_attn: torch.Tensor,
        mask: torch.Tensor,
        dropout_mask: torch.Tensor,
        p_dropout: float,
    ) -> torch.Tensor:
        """
        Backward pass for masked softmax with dropout.

        Args:
            grad_output: Gradient w.r.t. output [B, H, T, T]
            p_attn: Softmax output (before dropout) [B, H, T, T]
            mask: Attention mask [B, 1, T, T] (True=unmasked, False=masked)
            dropout_mask: Dropout mask [B, H, T, T] (True=kept, False=dropped)
            p_dropout: Dropout probability

        Returns:
            grad_scores: Gradient w.r.t. input scores [B, H, T, T]
        """
        if p_dropout > 0.0:
            grad_softmax_output = grad_output * dropout_mask.float() / (1.0 - p_dropout)
        else:
            grad_softmax_output = grad_output

        sum_term = (p_attn * grad_softmax_output).sum(dim=-1, keepdim=True)
        grad_softmax_input = p_attn * (grad_softmax_output - sum_term)
        grad_scores = grad_softmax_input.masked_fill(~mask, 0.0)

        return grad_scores


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
