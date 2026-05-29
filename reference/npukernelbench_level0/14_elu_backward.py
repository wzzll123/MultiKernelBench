import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Extended ELU backward.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, alpha: float, scale: float, input_scale: float, is_result: bool, self_or_result: torch.Tensor) -> torch.Tensor:
        # 保存原始dtype，如果输入是fp16，转为fp32计算以提升精度
        orig_dtype = grad_output.dtype
        if orig_dtype == torch.float16:
            grad_output = grad_output.float()
            self_or_result = self_or_result.float()
        
        mask = self_or_result <= 0
        if is_result:
            factor = torch.where(mask, input_scale * (self_or_result + alpha * scale), scale)
            result = grad_output * factor
        else:
            tmp = torch.exp(self_or_result * input_scale)
            factor = torch.where(mask, input_scale * alpha * scale * tmp, scale)
            result = grad_output * factor
        
        # 转回原始dtype
        if orig_dtype == torch.float16:
            result = result.to(orig_dtype)
        return result


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "14_elu_backward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        g_info = inputs[0]
        alpha_info = inputs[1]
        scale_info = inputs[2]
        input_scale_info = inputs[3]
        is_result_info = inputs[4]
        s_info = inputs[5]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        g_dtype = dtype_map[g_info["dtype"]]
        s_dtype = dtype_map[s_info["dtype"]]

        g = torch.randn(g_info["shape"], dtype=g_dtype)
        s = torch.randn(s_info["shape"], dtype=s_dtype)
        alpha = float(alpha_info["value"])
        scale = float(scale_info["value"])
        input_scale = float(input_scale_info["value"])
        is_result = bool(is_result_info["value"])
        input_groups.append([g, alpha, scale, input_scale, is_result, s])
    return input_groups


def get_init_inputs():
    return []
