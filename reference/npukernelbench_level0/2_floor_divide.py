import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs floor division with broadcasting support.
    torch.floor_divide(input, other) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies floor division to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): Dividend tensor (self).
            y (torch.Tensor): Divisor tensor (other), must be non-zero.

        Returns:
            torch.Tensor: Output tensor of floor division result.
        """
        # Handle bool inputs as aclnn executor does
        if x.dtype == torch.bool and y.dtype == torch.bool:
            x_ = x.to(torch.int64)
            y_ = y.to(torch.int64)
            return torch.floor_divide(x_, y_).to(torch.bool)
        elif x.dtype == torch.bool and y.dtype != torch.bool:
            x_ = x.to(y.dtype)
            return torch.floor_divide(x_, y)
        elif x.dtype != torch.bool and y.dtype == torch.bool:
            y_ = y.to(x.dtype)
            return torch.floor_divide(x, y_)
        else:
            return torch.floor_divide(x, y)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_floor_divide.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bool": torch.bool,
            "int32": torch.int32,
            "int8": torch.int8,
            "int16": torch.int16,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "float64": torch.float64,
        }
        x_dtype = dtype_map[x_info["dtype"]]
        y_dtype = dtype_map[y_info["dtype"]]

        def make_tensor(info, dtype, is_other=False):
            shape = info["shape"]
            if dtype == torch.bool:
                if is_other:
                    # Ensure divisor is non-zero (all True)
                    return torch.ones(shape, dtype=torch.bool)
                else:
                    return torch.randint(0, 2, shape, dtype=torch.bool)
            elif dtype == torch.uint8:
                if is_other:
                    # Non-zero: [1, 255]
                    return torch.randint(1, 256, shape, dtype=dtype)
                else:
                    return torch.randint(0, 256, shape, dtype=dtype)
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                if is_other:
                    # Non-zero: [-10, -1] U [1, 10]
                    t = torch.randint(1, 11, shape, dtype=dtype)
                    sign = torch.randint(0, 2, shape)
                    t = torch.where(sign.bool(), t, -t)
                    return t
                else:
                    return torch.randint(-10, 11, shape, dtype=dtype)
            else:
                # float types
                if is_other:
                    # Non-zero: avoid values close to zero
                    t = torch.randn(shape, dtype=dtype)
                    t = torch.where(t == 0, torch.ones_like(t), t)
                    return t
                else:
                    return torch.randn(shape, dtype=dtype)

        x = make_tensor(x_info, x_dtype, is_other=False)
        y = make_tensor(y_info, y_dtype, is_other=True)
        input_groups.append([x, y])
    return input_groups


def get_init_inputs():
    return []
