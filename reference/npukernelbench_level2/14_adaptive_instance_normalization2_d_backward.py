import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Instance Normalization 2D Backward Module.

    This module computes the backward pass (gradients) for Adaptive Instance Normalization 2D.
    It calculates gradients with respect to the input tensor, weight (gamma), and bias (beta)
    based on the gradient of the loss with respect to the output.

    The forward pass of AdaIN performs: output = weight * (x - mean) / std + bias
    This module computes the gradients needed for backpropagation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        Backward pass for Adaptive Instance Normalization 2D.

        Computes gradients with respect to input, weight (gamma), and bias (beta).

        Args:
            grad_output: Gradient of loss w.r.t. output, shape (N, C, H, W)
            x: Original input tensor from forward pass, shape (N, C, H, W)
            weight: Scale parameter (gamma), shape (C,)
            mean: Mean computed in forward pass, shape (N, C, 1, 1)
            std: Standard deviation computed in forward pass, shape (N, C, 1, 1)

        Returns:
            grad_input: Gradient w.r.t. input, shape (N, C, H, W)
            grad_weight: Gradient w.r.t. weight (gamma), shape (C,)
            grad_bias: Gradient w.r.t. bias (beta), shape (C,)
        """
        N, C, H, W = x.shape
        spatial_size = H * W
        x_centered = x - mean
        x_normalized = x_centered / std
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))
        weight_reshaped = weight.view(1, C, 1, 1)
        grad_output_scaled = grad_output * weight_reshaped
        grad_var = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
        grad_mean = grad_output_scaled.sum(dim=(2, 3), keepdim=True) * (-1.0 / std) + grad_var * (-2.0 * x_centered.mean(dim=(2, 3), keepdim=True))
        grad_input = grad_output_scaled / std + grad_var * 2.0 * x_centered / spatial_size + grad_mean / spatial_size
        return grad_input, grad_weight, grad_bias


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        idx = 0
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
                        if name == 'x':
                            if idx % 2 == 0:
                                mu = float(torch.empty(1).uniform_(-100, 100).item())
                                sigma = float(torch.empty(1).uniform_(1, 25).item())
                                tensors[name] = torch.normal(mu, sigma, shape, dtype=dtype) + torch.ones(shape, dtype=dtype)
                            else:
                                tensors[name] = torch.empty(shape, dtype=dtype).uniform_(-5, 5) + torch.ones(shape, dtype=dtype)
                        else:
                            tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
            idx += 1
    return input_groups


def get_init_inputs():
    return []
