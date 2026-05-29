import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that applies Extended ELU activation.
    ELU(x) = scale * x                    if x > 0
             alpha * scale * (exp(input_scale * x) - 1)  if x <= 0
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, alpha: float = 1.0, scale: float = 1.0, input_scale: float = 1.0) -> torch.Tensor:
        """
        Applies Extended ELU activation.

        Args:
            x (torch.Tensor): Input tensor.
            alpha (float, optional): Alpha parameter. Default: 1.0.
            scale (float, optional): Scale parameter. Default: 1.0.
            input_scale (float, optional): Input scale parameter. Default: 1.0.

        Returns:
            torch.Tensor: Output tensor.
        """
        alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
        scale_t = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
        input_scale_t = torch.as_tensor(input_scale, dtype=x.dtype, device=x.device)

        return torch.where(
            x > 0,
            scale_t * x,
            alpha_t * scale_t * (torch.exp(input_scale_t * x) - 1)
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_elu.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        alpha_info = inputs[1]
        scale_info = inputs[2]
        input_scale_info = inputs[3]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        alpha = float(alpha_info["value"])
        scale = float(scale_info["value"])
        input_scale = float(input_scale_info["value"])
        input_groups.append([x, alpha, scale, input_scale])
    return input_groups


def get_init_inputs():
    return []
