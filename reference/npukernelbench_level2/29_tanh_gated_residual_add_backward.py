import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Tanh-gated Residual Addition Backward Module.

    Backward pass for tanh-gated residual addition.
    Forward was: output = residual + tanh(gate) * hidden_states * mask

    Gradients:
    - grad_residual = grad_output (identity)
    - grad_hidden_states = grad_output * tanh(gate) * mask
    - grad_gate = sum(grad_output * hidden_states * mask * sech^2(gate))
              = sum(grad_output * hidden_states * mask * (1 - tanh^2(gate)))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, gate: torch.Tensor, hidden_states: torch.Tensor, mask: torch.Tensor):
        """
        Backward pass for tanh-gated residual addition.

        Args:
            grad_output: Gradient of loss w.r.t. output
            gate: Gate tensor from forward pass
            hidden_states: Hidden states from forward pass
            mask: Mask tensor from forward pass

        Returns:
            grad_residual: Gradient w.r.t. residual (same as grad_output)
            grad_hidden_states: Gradient w.r.t. hidden_states
            grad_gate: Gradient w.r.t. gate (scalar)
        """
        gate_float = gate.to(torch.float32)
        gate_value = torch.tanh(gate_float)
        grad_residual = grad_output.clone()
        grad_hidden_states = grad_output * gate_value * mask
        sech_squared = 1.0 - gate_value * gate_value
        masked_hidden_states = hidden_states * mask
        grad_gate = torch.sum(grad_output.to(torch.float32) * masked_hidden_states.to(torch.float32)) * sech_squared
        return grad_residual.to(torch.bfloat16), grad_hidden_states.to(torch.bfloat16), grad_gate.to(torch.bfloat16)


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
                    elif name == 'mask':
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randint(0, 2, shape, dtype=dtype)
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
