import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Embedding with Initial LayerNorm Backward Module.

    Backward pass for fused embedding + RMSNorm.
    Computes gradients w.r.t. embedding table and RMSNorm scale parameter.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.vocab_size = 65536
        self.hidden_size = 4096

    def forward(self, grad_output: torch.Tensor, input_ids: torch.Tensor, hidden_states_fp32: torch.Tensor, rstd: torch.Tensor, norm_weight: torch.Tensor):
        """
        Backward pass for fused embedding + RMSNorm.

        Args:
            grad_output: (batch_size, seq_len, hidden_size) gradient from next layer
            input_ids: (batch_size, seq_len) token indices
            hidden_states_fp32: (batch_size, seq_len, hidden_size) saved hidden states
            rstd: (batch_size, seq_len, 1) reciprocal standard deviation
            norm_weight: (hidden_size,) RMSNorm scale parameter

        Returns:
            grad_embed_weight: (vocab_size, hidden_size) gradient for embedding table
            grad_norm_weight: (hidden_size,) gradient for RMSNorm weight
        """
        batch_size, seq_len, _ = grad_output.shape
        grad_output_f32 = grad_output.to(torch.float32)
        normalized = hidden_states_fp32 * rstd
        grad_norm_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))
        grad_hidden = grad_output_f32 * norm_weight.to(torch.float32) * rstd
        grad_embed_weight = torch.zeros(self.vocab_size, self.hidden_size, dtype=torch.float32, device=grad_output.device)
        input_ids_flat = input_ids.reshape(-1)
        grad_hidden_flat = grad_hidden.reshape(-1, self.hidden_size)
        grad_embed_weight.index_add_(0, input_ids_flat, grad_hidden_flat)
        return grad_embed_weight.to(torch.bfloat16), grad_norm_weight.to(torch.bfloat16)


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
