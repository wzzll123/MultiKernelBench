import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that performs Group Normalization with Swish activation.
    torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0) -> (Tensor, Tensor, Tensor)
    Pure PyTorch implementation (replacing torch_npu.npu_group_norm_swish):

    def forward(self, input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0):
        N, C = input.shape[0], input.shape[1]
        # Reshape to [N, num_groups, C//num_groups, ...]
        input_reshaped = input.view(N, num_groups, C // num_groups, *input.shape[2:])

        # Compute mean and var over dims [2, 3, ...]
        dims = list(range(2, input_reshaped.ndim))
        mean = input_reshaped.mean(dim=dims, keepdim=True)
        var = input_reshaped.var(dim=dims, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (input_reshaped - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(input.shape)

        # Apply weight and bias
        if weight is not None:
            weight = weight.view(1, -1, *([1] * (input.ndim - 2)))
            x_norm = x_norm * weight
        if bias is not None:
            bias = bias.view(1, -1, *([1] * (input.ndim - 2)))
            x_norm = x_norm + bias

        # Swish activation: x * sigmoid(swish_scale * x)
        output = x_norm * torch.sigmoid(swish_scale * x_norm)

        # Compute mean and rstd for return
        mean_out = mean.view(N, num_groups)
        rstd = 1.0 / torch.sqrt(var + eps)
        rstd_out = rstd.view(N, num_groups)

        return output, mean_out, rstd_out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5, swish_scale: float = 1.0) -> tuple:
        """
        Applies group normalization followed by swish activation.

        Args:
            input (torch.Tensor): Input tensor for group normalization, supports 2-8D tensors.
                                  Supports float16, float32, bfloat16.
            num_groups (int): Number of groups to divide the first dimension into.
                              The first dimension must be divisible by num_groups.
            weight (torch.Tensor): Weight tensor, must be 1D with size equal to input's first dimension.
                                   Supports float16, float32, bfloat16, must match input dtype.
            bias (torch.Tensor): Bias tensor, must be 1D with size equal to input's first dimension.
                                 Supports float16, float32, bfloat16, must match input dtype.
            eps (float, optional): Value added to denominator for numerical stability. Default: 1e-5.
            swish_scale (float, optional): Scale value for swish computation. Default: 1.0.

        Returns:
            tuple: (output tensor, mean, rstd) where output is the normalized result with swish.
        """
        return torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_group_norm_swish.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        input_info = inputs[0]
        num_groups_info = inputs[1]
        weight_info = inputs[2]
        bias_info = inputs[3]
        eps_info = inputs[4]
        swish_scale_info = inputs[5]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[input_info["dtype"]]

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            inp = torch.normal(mu, sigma, input_info["shape"], dtype=dtype) + torch.ones(input_info["shape"], dtype=dtype)
        else:
            inp = torch.empty(input_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(input_info["shape"], dtype=dtype)
        num_groups = num_groups_info["value"]
        weight = torch.ones(weight_info["shape"], dtype=dtype)
        bias = torch.zeros(bias_info["shape"], dtype=dtype)
        eps = eps_info["value"]
        swish_scale = swish_scale_info["value"]
        input_groups.append([inp, num_groups, weight, bias, eps, swish_scale])
    return input_groups


def get_init_inputs():
    return []
