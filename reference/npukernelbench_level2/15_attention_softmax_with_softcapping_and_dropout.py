import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Attention Softmax with Softcapping and Dropout Module.

    Applies Gemma3's softcapping transformation followed by softmax normalization.
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30].
    """
    def __init__(self):
        super(Model, self).__init__()
        self.SOFTCAP = 30.0

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply softcapping transformation followed by softmax.

        Args:
            attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        scaled = attn_weights / self.SOFTCAP
        clamped = torch.tanh(scaled)
        softcapped = clamped * self.SOFTCAP
        output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        return output


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
