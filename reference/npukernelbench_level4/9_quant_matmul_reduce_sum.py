import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs quantized matmul reduce sum using NPU accelerated npu_quant_matmul_reduce_sum.
    Computes quantized grouped matrix multiplication and sums results across all groups.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, *,
                x1_scale=None, x2_scale=None):
        """
        Performs quantized matmul reduce sum on NPU.

        Args:
            x1 (Tensor): Left matrix, shape [batch, m, k], dtype int8.
            x2 (Tensor): Right matrix in NZ format, shape [batch, k, n], dtype int8.
            x1_scale (Tensor): Left matrix quantization scale, shape [batch, m], dtype float32.
            x2_scale (Tensor): Right matrix quantization scale, shape [n], dtype bfloat16.

        Returns:
            Tensor: Sum of all group matmul results, shape [m, n], dtype bfloat16.
        """
        import torch_npu
        return torch_npu.npu_quant_matmul_reduce_sum(x1, x2,
                                                      x1_scale=x1_scale,
                                                      x2_scale=x2_scale)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_quant_matmul_reduce_sum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x1_info = inputs[0]
        x2_info = inputs[1]
        x1_scale_info = inputs[2]
        x2_scale_info = inputs[3]

        x1 = torch.randint(-5, 5, x1_info["shape"], dtype=torch.int8)
        x2 = torch.randint(-5, 5, x2_info["shape"], dtype=torch.int8)
        x1_scale = torch.randn(x1_scale_info["shape"], dtype=torch.float32)
        x2_scale = torch.randn(x2_scale_info["shape"], dtype=torch.bfloat16)

        input_groups.append([x1, x2, x1_scale, x2_scale])
    return input_groups


def get_init_inputs():
    return []
