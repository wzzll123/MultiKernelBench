import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Element-wise real division using PyTorch.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: float, fusion_mode: str = 'float32') -> torch.Tensor:
        return torch.div(x1, x2)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_real_div.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        fusion_mode_info = inputs[2]

        x1 = torch.randn(x1_info["shape"], dtype=torch.float32)
        x2 = float(x2_info["value"])
        fusion_mode = fusion_mode_info["value"]
        input_groups.append([x1, x2, fusion_mode])
    return input_groups


def get_init_inputs():
    return []
