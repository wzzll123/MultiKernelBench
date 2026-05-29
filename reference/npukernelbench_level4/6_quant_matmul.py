import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs quantized matrix multiplication using NPU accelerated npu_quant_matmul.
    Supports int8/int4 quantized matmul with various output data types.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor, *,
                offset=None, pertoken_scale=None, bias=None, output_dtype=None, group_sizes=None):
        """
        Performs quantized matmul on NPU.

        Args:
            x1 (Tensor): Left matrix, shape 2-6D, dtype int8/int32.
            x2 (Tensor): Right matrix, shape 2-6D, dtype int8/int32.
            scale (Tensor): Quantization scale factor, dtype float32/int64/bfloat16.
            offset (Tensor, optional): Quantization offset, dtype float32.
            pertoken_scale (Tensor, optional): Per-token scale, shape [m], dtype float32.
            bias (Tensor, optional): Bias term, shape [n] or [batch, 1, n], dtype int32/bfloat16/float32.
            output_dtype (int, optional): Output data type, supports int8/float16/bfloat16/int32.
            group_sizes (list[int], optional): Group quantization granularity.

        Returns:
            Tensor: Quantized matmul result.
        """
        import torch_npu
        return torch_npu.npu_quant_matmul(x1, x2, scale,
                                           offset=offset, pertoken_scale=pertoken_scale,
                                           bias=bias, output_dtype=output_dtype,
                                           group_sizes=group_sizes)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_quant_matmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x1_info = inputs[0]
        x2_info = inputs[1]
        scale_info = inputs[2]

        x1 = torch.randint(-5, 5, x1_info["shape"], dtype=torch.int8)
        x2 = torch.randint(-5, 5, x2_info["shape"], dtype=torch.int8)

        scale_dtype_map = {
            "float32": torch.float32,
            "int64": torch.int64,
            "bfloat16": torch.bfloat16,
        }
        scale_dtype = scale_dtype_map.get(scale_info["dtype"], torch.float32)
        scale = torch.randn(scale_info["shape"], dtype=scale_dtype)

        offset = None
        pertoken_scale = None
        bias = None
        output_dtype = None
        group_sizes = None

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "offset":
                offset_dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                }
                offset_dtype = offset_dtype_map.get(inp.get("dtype", "float32"), torch.float32)
                offset = torch.randn(inp["shape"], dtype=offset_dtype)
            elif name == "pertoken_scale":
                pertoken_scale = torch.randn(inp["shape"], dtype=torch.float32)
            elif name == "bias":
                bias_dtype_map = {
                    "int32": torch.int32,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "float16": torch.float16,
                }
                bias_dtype = bias_dtype_map.get(inp.get("dtype", "int32"), torch.int32)
                bias = torch.randint(-5, 5, inp["shape"], dtype=bias_dtype)
            elif name == "output_dtype":
                output_dtype = inp["value"]
            elif name == "group_sizes":
                group_sizes = inp.get("value")

        input_groups.append([x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes])
    return input_groups


def get_init_inputs():
    return []
