import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs standard 2D convolution.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) -> torch.Tensor:
        """
        Applies standard 2D convolution to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel (will be converted to (kernel_size, kernel_size)).
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True.

        Returns:
            torch.Tensor: Output tensor after performing nn.Conv2d.
        """
        conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        return conv(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_conv_standard2d.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        
        tensor_inputs = [inp for inp in inputs if inp.get("type") == "tensor"]
        attr_inputs = {inp["name"]: inp["value"] for inp in inputs if inp.get("type") == "attr"}
        
        x_info = tensor_inputs[0]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(x_info["dtype"], torch.float32)
        x = torch.randn(x_info["shape"], dtype=dtype)
        
        in_channels = attr_inputs.get("in_channels")
        out_channels = attr_inputs.get("out_channels")
        kernel_size = attr_inputs.get("kernel_size")
        stride = attr_inputs.get("stride", 1)
        padding = attr_inputs.get("padding", 0)
        dilation = attr_inputs.get("dilation", 1)
        groups = attr_inputs.get("groups", 1)
        bias = attr_inputs.get("bias", True)
        
        input_groups.append([x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias])
    return input_groups


def get_init_inputs():
    return []
