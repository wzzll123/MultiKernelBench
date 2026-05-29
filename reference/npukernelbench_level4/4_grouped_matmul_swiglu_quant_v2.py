import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs Grouped Matmul + SwiGLU + Quant computation using NPU accelerated npu_grouped_matmul_swiglu_quant_v2.
    Fuses grouped matrix multiplication, SwiGLU activation, and quantization.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor, *,
                smooth_scale=None, weight_assist_matrix=None, bias=None,
                dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                group_list_type=0, tuning_config=None):
        """
        Performs grouped matmul + SwiGLU + quant on NPU.

        Args:
            x (Tensor): Left matrix for matmul, shape [m, k], dtype int8.
            weight (TensorList): Weight matrices, shape [e, k, n], dtype int8/int32.
            weight_scale (TensorList): Weight quantization scales, dtype float32.
            x_scale (Tensor): Left matrix quantization scale, shape [m], dtype float32.
            group_list (Tensor): Token count per group, shape [e], dtype int64.
            smooth_scale (Tensor, optional): Quantization smooth scales, dtype float32.
            weight_assist_matrix (TensorList, optional): Weight assist matrix, dtype float32.
            bias (Tensor, optional): Matmul bias, shape 2D, dtype int32.
            dequant_mode (int): Dequantization mode, 0=left pertoken right perchannel, 1=left pertoken right pergroup.
            dequant_dtype (int): Dequantization dtype, reserved, default 0.
            quant_mode (int): Quantization mode after SwiGLU, 0=pertoken, 1=perchannel.
            quant_dtype (int): Quantized low-bit dtype, 0=int8, 1=float8_e8m0, 2=float8_e5m2, 3=float8_e4m3.
            group_list_type (int): Group list input type, 0=cumsum, 1=count.
            tuning_config (List[int], optional): Tuning configuration.

        Returns:
            tuple: (output, output_scale) where output is int8 [m, n] and output_scale is float [m].
        """
        import torch_npu
        return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, weight, weight_scale, x_scale, group_list,
            smooth_scale=smooth_scale, weight_assist_matrix=weight_assist_matrix,
            bias=bias, dequant_mode=dequant_mode, dequant_dtype=dequant_dtype,
            quant_mode=quant_mode, quant_dtype=quant_dtype,
            group_list_type=group_list_type, tuning_config=tuning_config)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_grouped_matmul_swiglu_quant_v2.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x_info = inputs[0]
        weight_info = inputs[1]
        weight_scale_info = inputs[2]
        x_scale_info = inputs[3]
        group_list_info = inputs[4]

        x = torch.randint(-128, 127, x_info["shape"], dtype=torch.int8)
        weight = [torch.randint(-128, 127, weight_info["shape"], dtype=torch.int8)]
        weight_scale = [torch.randn(weight_scale_info["shape"], dtype=torch.float32)]
        x_scale = torch.randn(x_scale_info["shape"], dtype=torch.float32)
        group_list = torch.tensor(group_list_info["value"], dtype=torch.int64)

        smooth_scale = None
        weight_assist_matrix = None
        bias = None
        dequant_mode = 0
        dequant_dtype = 0
        quant_mode = 0
        quant_dtype = 0
        group_list_type = 0
        tuning_config = None

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "smooth_scale":
                smooth_scale = None
            elif name == "weight_assist_matrix":
                weight_assist_matrix = None
            elif name == "bias":
                bias = None
            elif name == "dequant_mode":
                dequant_mode = inp["value"]
            elif name == "dequant_dtype":
                dequant_dtype = inp["value"]
            elif name == "quant_mode":
                quant_mode = inp["value"]
            elif name == "quant_dtype":
                quant_dtype = inp["value"]
            elif name == "group_list_type":
                group_list_type = inp["value"]
            elif name == "tuning_config":
                tuning_config = inp.get("value")

        input_groups.append([x, weight, weight_scale, x_scale, group_list,
                             smooth_scale, weight_assist_matrix, bias,
                             dequant_mode, dequant_dtype, quant_mode, quant_dtype,
                             group_list_type, tuning_config])
    return input_groups


def get_init_inputs():
    return []
