import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Hardsigmoid activation function.
    torch.nn.Hardsigmoid()(input) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()
        self.m = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Hardsigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Hardsigmoid activation.
        """
        if x.dtype == torch.int32:
            x = x.to(torch.float32)
            return self.m(x).to(torch.int32)
        return self.m(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_hardsigmoid.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
        }
        dtype = dtype_map[x_info["dtype"]]
        if dtype == torch.int32:
            x = torch.randint(-100, 101, x_info["shape"], dtype=dtype)
        else:
            x = torch.randn(x_info["shape"], dtype=dtype)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
