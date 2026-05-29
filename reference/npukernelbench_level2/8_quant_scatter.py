import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs quantized scatter operation.
    torch_npu.npu_quant_scatter(input, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1, reduce='update') -> Tensor
    PyTorch native implementation of forward function
    def forward(self, input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor,
                quant_scales: torch.Tensor, quant_zero_points: torch.Tensor = None,
                axis: int = 0, quant_axis: int = 1, reduce: str = 'update') -> torch.Tensor:
        if axis < 0:
            axis = input.ndim + axis

        output = input.clone()

        neg_inf_mask = (updates == -torch.inf)
        pos_inf_mask = (updates == torch.inf)

        quant_scales_expanded = quant_scales
        while quant_scales_expanded.ndim < updates.ndim:
            quant_scales_expanded = quant_scales_expanded.unsqueeze(0)

        if quant_zero_points is not None:
            quant_zp_expanded = quant_zero_points
            while quant_zp_expanded.ndim < updates.ndim:
                quant_zp_expanded = quant_zp_expanded.unsqueeze(0)
            quantized = torch.round(updates.float() / quant_scales_expanded.float() + quant_zp_expanded.float())
        else:
            quantized = torch.round(updates.float() / quant_scales_expanded.float())
        quantized = quantized.clamp(-128, 127).to(torch.int8)

        quantized[neg_inf_mask] = -128
        quantized[pos_inf_mask] = 127

        indices_int64 = indices.to(torch.int64)
        update_len = updates.shape[axis]

        for i in range(indices_int64.shape[0]):
            idx_val = indices_int64[i].item()
            for j in range(update_len):
                src_slices = [slice(None)] * quantized.ndim
                src_slices[0] = i
                src_slices[axis] = j

                dst_slices = [slice(None)] * output.ndim
                dst_slices[0] = i
                dst_slices[axis] = idx_val + j

                output[tuple(dst_slices)] = quantized[tuple(src_slices)]

        return output
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor,
                quant_scales: torch.Tensor, quant_zero_points: torch.Tensor = None,
                axis: int = 0, quant_axis: int = 1, reduce: str = 'update') -> torch.Tensor:
        """
        Performs quantized scatter operation.

        Args:
            input (torch.Tensor): Source data tensor. dtype: int8, format: ND.
                                  Supports non-contiguous tensors. Must be 3-8D.
            indices (torch.Tensor): Index tensor. dtype: int32, format: ND.
                                    Supports non-contiguous tensors.
            updates (torch.Tensor): Update data tensor. format: ND, supports non-contiguous.
                                    dtype: bfloat16, float16.
            quant_scales (torch.Tensor): Quantization scale tensor. format: ND, supports non-contiguous.
                                         dtype: bfloat16, float32.
            quant_zero_points (torch.Tensor, optional): Quantization offset tensor. format: ND.
                                                        dtype: bfloat16, int32.
            axis (int, optional): Axis on updates for updating. Default: 0.
            quant_axis (int, optional): Axis on updates for quantization. Default: 1.
            reduce (str, optional): Data operation mode. Currently only supports 'update'. Default: 'update'.

        Returns:
            torch.Tensor: Output tensor after quantized scatter operation.
        """
        return torch_npu.npu_quant_scatter(input, indices, updates, quant_scales,
                                           quant_zero_points=quant_zero_points, axis=axis,
                                           quant_axis=quant_axis, reduce=reduce)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 1000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
