import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs dynamic quantization on NPU.
    Pytorch native implemention
    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None,
                group_index: torch.Tensor = None, dst_type=None):
        if dst_type is None:
            dst_type = torch.int8

        x_float = x.float()

        if smooth_scales is not None:
            smooth_scales_float = smooth_scales.float()
            x_float = x_float * smooth_scales_float

        if group_index is not None:
            return self._quant_with_groups(x_float, group_index, dst_type)

        return self._quant_per_token(x_float, dst_type)

    def _quant_per_token(self, x: torch.Tensor, dst_type):
        if x.dim() == 2:
            max_abs = x.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            scale = scale.clamp(min=1e-10)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            scale = scale.squeeze(1)
            return quantized, scale
        elif x.dim() == 3:
            max_abs = x.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs / 127.0
            scale = scale.clamp(min=1e-10)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            scale = scale.squeeze(2)
            return quantized, scale
        else:
            max_abs = x.abs().max()
            scale = max_abs / 127.0
            scale = torch.tensor(scale, device=x.device)
            quantized = torch.round(x / scale)
            quantized = quantized.clamp(-128, 127).to(dst_type)
            return quantized, scale

    def _quant_with_groups(self, x: torch.Tensor, group_index: torch.Tensor, dst_type):
        if x.dim() != 2:
            raise ValueError("Group quantization only supports 2D tensors")

        num_tokens = x.shape[0]
        quantized = torch.zeros_like(x, dtype=dst_type)
        scales = torch.zeros(num_tokens, device=x.device)

        num_groups = group_index.max().item() + 1 if group_index.numel() > 0 else 1

        for g in range(num_groups):
            mask = (group_index == g)
            if mask.sum() == 0:
                continue

            group_x = x[mask]
            max_abs = group_x.abs().max()
            scale = max_abs / 127.0
            scale = max(scale, 1e-10)

            group_quantized = torch.round(group_x / scale)
            group_quantized = group_quantized.clamp(-128, 127).to(dst_type)

            quantized[mask] = group_quantized
            scales[mask] = scale

        return quantized, scales
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None,
                group_index: torch.Tensor = None, dst_type=None):
        """
        Performs dynamic quantization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be quantized.
            smooth_scales (torch.Tensor, optional): Smooth scale factors.
            group_index (torch.Tensor, optional): Group indices for per-group quantization.
            dst_type (optional): Target data type for quantized output.

        Returns:
            tuple: (quantized_tensor, scale_tensor)
        """
        import torch_npu
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales,
                                            group_index=group_index, dst_type=dst_type)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "29_dynamic_quant.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
