import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs padding on a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, pad: tuple, mode: str = 'constant', value: float = None) -> torch.Tensor:
        """
        Pads tensor with specified padding mode.

        Args:
            x (torch.Tensor): Input tensor.
            pad (tuple): Padding sizes in the form (pad_left, pad_right, pad_top, pad_bottom, ...).
            mode (str, optional): Padding mode: 'constant', 'reflect', 'replicate', 'circular'.
            value (float, optional): Fill value for 'constant' padding.

        Returns:
            torch.Tensor: Padded tensor.
        """
        return torch.nn.functional.pad(x, pad, mode=mode, value=value)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_pad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        
        pad = None
        mode = "constant"
        value = None
        
        for inp in inputs[1:]:
            if inp["name"] == "pad":
                pad = tuple(inp["value"])
            elif inp["name"] == "mode":
                mode = inp["value"]
            elif inp["name"] == "value":
                value = inp["value"]
        
        input_groups.append([x, pad, mode, value])
    return input_groups


def get_init_inputs():
    return []
