import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs weight-quantized batch matrix multiplication using NPU accelerated npu_weight_quant_batchmatmul.
    Supports pertensor, perchannel, and pergroup quantization for weight matrices.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                antiquant_scale: torch.Tensor, antiquant_offset=None,
                quant_scale=None, quant_offset=None, bias=None,
                antiquant_group_size=0, inner_precise=0):
        """
        Performs weight-quantized batch matmul on NPU.

        Args:
            x (Tensor): Left matrix, shape [M, K], dtype float16/bfloat16.
            weight (Tensor): Right matrix (weight), shape [K, N], dtype int8/int32.
            antiquant_scale (Tensor): Dequantization scale for weight, dtype float16/bfloat16/int64.
            antiquant_offset (Tensor, optional): Dequantization offset for weight, dtype float16/bfloat16/int32.
            quant_scale (Tensor, optional): Output quantization scale, dtype float32/int64.
            quant_offset (Tensor, optional): Output quantization offset, dtype float32.
            bias (Tensor, optional): Bias term, shape [1, N] or [N], dtype float16/float32.
            antiquant_group_size (int): Group size for pergroup quantization, default 0.
            inner_precise (int): 0=high precision, 1=high performance, default 0.

        Returns:
            Tensor: Output tensor. int8 if quant_scale provided, otherwise same dtype as x.
        """
        import torch_npu
        return torch_npu.npu_weight_quant_batchmatmul(
            x, weight, antiquant_scale, antiquant_offset,
            quant_scale, quant_offset, bias, antiquant_group_size, inner_precise)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_weight_quant_batchmatmul.json")
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
        weight_info = inputs[1]

        dtype = dtype_map[x_info["dtype"]]
        weight_dtype = dtype_map.get(weight_info.get("dtype", "float16"), dtype)

        x = torch.randn(x_info["shape"], dtype=dtype)
        weight = torch.randint(-8, 8, weight_info["shape"], dtype=torch.int8) if weight_dtype != dtype else torch.randn(weight_info["shape"], dtype=dtype)

        antiquant_scale = None
        antiquant_offset = None
        quant_scale = None
        quant_offset = None
        bias = None
        antiquant_group_size = 0
        inner_precise = 0

        for inp in inputs[2:]:
            name = inp.get("name", "")
            if name == "antiquant_scale":
                aq_scale_dtype = dtype_map.get(inp.get("dtype", "float16"), dtype)
                antiquant_scale = torch.randn(inp["shape"], dtype=aq_scale_dtype)
            elif name == "antiquant_offset":
                aq_offset_dtype = dtype_map.get(inp.get("dtype", "float16"), dtype)
                antiquant_offset = torch.randn(inp["shape"], dtype=aq_offset_dtype)
            elif name == "quant_scale":
                quant_scale = torch.randn(inp["shape"], dtype=torch.float32)
            elif name == "quant_offset":
                quant_offset = torch.randn(inp["shape"], dtype=torch.float32)
            elif name == "bias":
                bias_dtype = dtype_map.get(inp.get("dtype", "float16"), dtype)
                bias = torch.randn(inp["shape"], dtype=bias_dtype)
            elif name == "antiquant_group_size":
                antiquant_group_size = inp["value"]
            elif name == "inner_precise":
                inner_precise = inp["value"]

        if antiquant_scale is None:
            antiquant_scale = torch.ones([1], dtype=dtype)

        input_groups.append([x, weight, antiquant_scale, antiquant_offset,
                             quant_scale, quant_offset, bias,
                             antiquant_group_size, inner_precise])
    return input_groups


def get_init_inputs():
    return []
