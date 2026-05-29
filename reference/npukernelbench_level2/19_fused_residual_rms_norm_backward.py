import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Backward pass for fused residual addition and RMSNorm.
    Computes gradients for hidden_states, residual, and weight given the gradient of the output.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        normalized: torch.Tensor,
        rstd: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple:
        """
        Computes gradients for the fused residual + RMSNorm backward pass.

        Args:
            grad_output (torch.Tensor): Gradient from next layer with shape [batch_size, seq_len, hidden_size].
                                        Supports bfloat16.
            x (torch.Tensor): Saved tensor: hidden_states + residual from forward pass
                              with shape [batch_size, seq_len, hidden_size]. Supports float32.
            normalized (torch.Tensor): Saved tensor: normalized values from forward pass
                                       with shape [batch_size, seq_len, hidden_size]. Supports float32.
            rstd (torch.Tensor): Saved tensor: reciprocal standard deviation from forward pass
                                 with shape [batch_size, seq_len, 1]. Supports float32.
            weight (torch.Tensor): RMSNorm scale parameter with shape [hidden_size]. Supports float32.

        Returns:
            tuple: (grad_hidden_states, grad_residual, grad_weight)
                - grad_hidden_states (torch.Tensor): Gradient w.r.t. hidden_states
                                                     with shape [batch_size, seq_len, hidden_size].
                - grad_residual (torch.Tensor): Gradient w.r.t. residual
                                                with shape [batch_size, seq_len, hidden_size].
                - grad_weight (torch.Tensor): Gradient w.r.t. weight with shape [hidden_size].
        """
        grad_output_f32 = grad_output.to(torch.float32)

        grad_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))

        grad_normalized = grad_output_f32 * weight

        mean_grad_norm = (grad_normalized * normalized).mean(dim=-1, keepdim=True)

        grad_x = rstd * (grad_normalized - mean_grad_norm * normalized)

        grad_x_bf16 = grad_x.to(torch.bfloat16)

        grad_hidden_states = grad_x_bf16
        grad_residual = grad_x_bf16.clone()

        return grad_hidden_states, grad_residual, grad_weight


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
