import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs 3D max pooling.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                dilation: int = 1, ceil_mode: bool = False,
                return_indices: bool = False):
        """
        Applies 3D max pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            dilation (int, optional): Spacing between kernel elements.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            return_indices (bool, optional): Return indices of max values.

        Returns:
            torch.Tensor or tuple: Pooled tensor (and indices if return_indices=True).
        """
        return torch.nn.functional.max_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode,
            return_indices=return_indices
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "27_max_pool3d.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        kernel_size_info = inputs[1]
        stride_info = inputs[2]
        padding_info = inputs[3]
        dilation_info = inputs[4]
        ceil_mode_info = inputs[5]
        return_indices_info = inputs[6]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        kernel_size = kernel_size_info["value"]
        stride = stride_info["value"]
        padding = padding_info["value"]
        dilation = dilation_info["value"]
        ceil_mode = ceil_mode_info["value"]
        return_indices = return_indices_info["value"]
        input_groups.append([x, kernel_size, stride, padding, dilation, ceil_mode, return_indices])
    return input_groups


def get_init_inputs():
    return []
