import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs FFN (Feed-Forward Network) computation using NPU accelerated npu_ffn.
    out = activation(x * W1 + b1) * W2 + b2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                activation: str, *, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None,
                deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None,
                antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=None, output_dtype=None):
        """
        Performs FFN computation on NPU.

        Args:
            x (Tensor): Input tensor of shape [M, K1] (or up to 8D).
            weight1 (Tensor): Weight for first matmul, shape [K1, N1] or [E, K1, N1].
            weight2 (Tensor): Weight for second matmul, shape [K2, N2] or [E, K2, N2].
            activation (str): Activation function, supports fastgelu/gelu/relu/silu/geglu/swiglu/reglu.
            expert_tokens (list, optional): Token count per expert.
            expert_tokens_index (list, optional): Token index per expert.
            bias1 (Tensor, optional): Bias for first matmul.
            bias2 (Tensor, optional): Bias for second matmul.
            scale (Tensor, optional): Quantization scale.
            offset (Tensor, optional): Quantization offset.
            deq_scale1 (Tensor, optional): Dequantization scale for first matmul.
            deq_scale2 (Tensor, optional): Dequantization scale for second matmul.
            antiquant_scale1 (Tensor, optional): Anti-quantization scale for first matmul.
            antiquant_scale2 (Tensor, optional): Anti-quantization scale for second matmul.
            antiquant_offset1 (Tensor, optional): Anti-quantization offset for first matmul.
            antiquant_offset2 (Tensor, optional): Anti-quantization offset for second matmul.
            inner_precise (int, optional): 0 for high precision, 1 for high performance.
            output_dtype (ScalarType, optional): Output data type for quantization scenario.

        Returns:
            Tensor: Output tensor with same dimensions as x.
        """
        import torch_npu
        return torch_npu.npu_ffn(x, weight1, weight2, activation,
                                  expert_tokens=expert_tokens,
                                  expert_tokens_index=expert_tokens_index,
                                  bias1=bias1, bias2=bias2,
                                  scale=scale, offset=offset,
                                  deq_scale1=deq_scale1, deq_scale2=deq_scale2,
                                  antiquant_scale1=antiquant_scale1,
                                  antiquant_scale2=antiquant_scale2,
                                  antiquant_offset1=antiquant_offset1,
                                  antiquant_offset2=antiquant_offset2,
                                  inner_precise=inner_precise,
                                  output_dtype=output_dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_ffn.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }

        x_info = inputs[0]
        weight1_info = inputs[1]
        weight2_info = inputs[2]
        activation_info = inputs[3]

        dtype = dtype_map[x_info["dtype"]]

        x = torch.randn(x_info["shape"], dtype=dtype)
        weight1 = torch.randn(weight1_info["shape"], dtype=dtype)
        weight2 = torch.randn(weight2_info["shape"], dtype=dtype)
        activation = activation_info["value"]

        expert_tokens = None
        expert_tokens_index = None
        bias1 = None
        bias2 = None
        scale = None
        offset = None
        deq_scale1 = None
        deq_scale2 = None
        antiquant_scale1 = None
        antiquant_scale2 = None
        antiquant_offset1 = None
        antiquant_offset2 = None
        inner_precise = None
        output_dtype = None

        for inp in inputs[4:]:
            name = inp.get("name", "")
            if name == "expert_tokens":
                expert_tokens = inp.get("value")
            elif name == "expert_tokens_index":
                expert_tokens_index = inp.get("value")
            elif name == "bias1":
                bias1 = torch.randn(inp["shape"], dtype=dtype_map.get(inp["dtype"], dtype))
            elif name == "bias2":
                bias2 = torch.randn(inp["shape"], dtype=dtype_map.get(inp["dtype"], dtype))
            elif name == "scale":
                scale = torch.randn(inp["shape"], dtype=torch.float32)
            elif name == "offset":
                offset = torch.randn(inp["shape"], dtype=torch.float32)
            elif name == "deq_scale1":
                deq_scale1 = torch.randn(inp["shape"], dtype=dtype_map.get(inp["dtype"], torch.float32))
            elif name == "deq_scale2":
                deq_scale2 = torch.randn(inp["shape"], dtype=dtype_map.get(inp["dtype"], torch.float32))
            elif name == "antiquant_scale1":
                antiquant_scale1 = torch.randn(inp["shape"], dtype=dtype)
            elif name == "antiquant_scale2":
                antiquant_scale2 = torch.randn(inp["shape"], dtype=dtype)
            elif name == "antiquant_offset1":
                antiquant_offset1 = torch.randn(inp["shape"], dtype=dtype)
            elif name == "antiquant_offset2":
                antiquant_offset2 = torch.randn(inp["shape"], dtype=dtype)
            elif name == "inner_precise":
                inner_precise = inp["value"]
            elif name == "output_dtype":
                output_dtype = inp["value"]

        input_groups.append([x, weight1, weight2, activation,
                             expert_tokens, expert_tokens_index,
                             bias1, bias2, scale, offset,
                             deq_scale1, deq_scale2,
                             antiquant_scale1, antiquant_scale2,
                             antiquant_offset1, antiquant_offset2,
                             inner_precise, output_dtype])
    return input_groups


def get_init_inputs():
    return []
