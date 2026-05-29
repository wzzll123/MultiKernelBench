import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs element-wise logical AND with broadcasting support.
    torch.logical_and(input, other) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise logical AND to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.

        Returns:
            torch.Tensor: Output bool tensor of the logical AND result.
        """
        return torch.logical_and(x, y)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_logical_and.json")
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
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        x_dtype = dtype_map[x_info["dtype"]]
        y_dtype = dtype_map[y_info["dtype"]]

        def make_tensor(info, dtype):
            shape = info["shape"]
            if dtype == torch.bool:
                return torch.randint(0, 2, shape, dtype=torch.bool)
            elif dtype == torch.uint8:
                return torch.randint(0, 10, shape, dtype=dtype)
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                return torch.randint(-10, 10, shape, dtype=dtype)
            elif dtype in (torch.complex64, torch.complex128):
                real = torch.randn(shape)
                imag = torch.randn(shape)
                return torch.complex(real, imag).to(dtype)
            else:
                return torch.randn(shape, dtype=dtype)

        x = make_tensor(x_info, x_dtype)
        y = make_tensor(y_info, y_dtype)
        input_groups.append([x, y])
    return input_groups


def get_init_inputs():
    return []
