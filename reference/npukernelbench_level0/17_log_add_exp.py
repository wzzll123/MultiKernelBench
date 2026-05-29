import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Element-wise logaddexp.
    torch.logaddexp(input, other) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise logaddexp.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_is_f64 = x.dtype == torch.float64
        y_is_f64 = y.dtype == torch.float64
        if not (x_is_f64 or y_is_f64):
            return torch.logaddexp(x, y)

        if x_is_f64 or y_is_f64:
            out_dtype = torch.float32
        else:
            out_dtype = x.dtype if x.dtype == y.dtype else torch.float32
        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        result = torch.empty(out_shape, dtype=out_dtype, device=x.device)
        torch.logaddexp(x, y, out=result)
        return result


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "17_log_add_exp.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]

        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp64": torch.float64,
            "float64": torch.float64,
        }
        x_dtype = dtype_map[x_info["dtype"]]
        y_dtype = dtype_map[y_info["dtype"]]

        if x_dtype == torch.float16:
            x_dtype = torch.float32
        
        if y_dtype == torch.float16:
            y_dtype = torch.float32
        
        x_shape = x_info["shape"]
        y_shape = y_info["shape"]

        x = torch.randn(x_shape, dtype=x_dtype)
        y = torch.randn(y_shape, dtype=y_dtype)
        input_groups.append([x, y])
    return input_groups


def get_init_inputs():
    return []
