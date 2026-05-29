import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs interpolation (resizing) of tensors.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, size=None, scale_factor=None,
                mode: str = 'nearest', align_corners=None,
                recompute_scale_factor=None, antialias: bool = False) -> torch.Tensor:
        """
        Interpolates (resizes) the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, ...) where ... represents spatial dimensions.
            size (optional): Output spatial size.
            scale_factor (optional): Multiplier for spatial size.
            mode (str, optional): Algorithm used for interpolation: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'.
            align_corners (optional): How to align corners when resizing.
            recompute_scale_factor (optional): Recompute scale_factor for backward compatibility.
            antialias (bool, optional): Apply antialiasing.

        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return torch.nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "28_interpolate.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(x_info["shape"], dtype=dtype)
        
        size = None
        scale_factor = None
        mode = "nearest"
        align_corners = None
        
        for inp in inputs[1:]:
            if inp["name"] == "size":
                size = inp["value"]
            elif inp["name"] == "scale_factor":
                scale_factor = inp["value"]
            elif inp["name"] == "mode":
                mode = inp["value"]
            elif inp["name"] == "align_corners":
                align_corners = inp["value"]
        
        input_groups.append([x, size, scale_factor, mode, align_corners])
    return input_groups


def get_init_inputs():
    return []
