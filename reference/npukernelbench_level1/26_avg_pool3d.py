import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs 3D average pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                ceil_mode: bool = False, count_include_pad: bool = True,
                divisor_override=None) -> torch.Tensor:
        """
        Applies 3D average pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            count_include_pad (bool, optional): Include zero-padding in averaging.
            divisor_override (optional): If specified, will be used as divisor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return torch.nn.functional.avg_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "26_avg_pool3d.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        kernel_size_info = inputs[1]
        stride_info = inputs[2]
        padding_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
        kernel_size = kernel_size_info["value"]
        stride = stride_info["value"]
        padding = padding_info["value"]
        input_groups.append([x, kernel_size, stride, padding])
    return input_groups


def get_init_inputs():
    return []
