import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs SwiGLU with quantization.
    torch_npu.npu_swiglu_quant(x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0, group_list_type=0, dst_type=None) -> (Tensor, Tensor)
    PyTorch native implementation of forward function
    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None, offsets: torch.Tensor = None,    
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                group_list_type: int = 0, dst_type = None) -> tuple:
        x_float = x.float()

        half_size = x_float.shape[-1] // 2
        x_left = x_float[..., :half_size]
        x_right = x_float[..., half_size:]

        if activate_left:
            activated = torch.sigmoid(x_left) * x_left
            output = activated * x_right
        else:
            activated = torch.sigmoid(x_right) * x_right
            output = activated * x_left

        # Reshape to 2D for group operations
        ori_shape = output.shape
        prefix_shape = output.shape[:-1]
        prefix_dims = 1
        for s in prefix_shape:
            prefix_dims *= s
        last_dim = output.shape[-1]
        output_2d = output.reshape(prefix_dims, last_dim)
        swiglu_2d = output_2d.clone()

        # Parse group boundaries
        boundaries = None
        if group_index is not None:
            if group_list_type == 0:
                boundaries = [0] + group_index.tolist()
            else:
                boundaries = [0]
                cumsum = 0
                for c in group_index.tolist():
                    cumsum += int(c)
                    boundaries.append(cumsum)

        # Determine int scale based on dst_type
        is_int4 = dst_type is not None and (str(dst_type) == 'int4' or dst_type == torch.quint4x2)
        if is_int4:
            int_scale = 7
            clip_min, clip_max = -8, 7
        else:
            int_scale = 127
            clip_min, clip_max = -128, 127

        if quant_mode == 1:  # Dynamic quantization
            # Apply group scales
            if smooth_scales is not None and boundaries is not None:
                smooth_f = smooth_scales.float()
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    if start < end and i < smooth_f.shape[0]:
                        s = smooth_f[i]
                        if s.dim() >= 1 and s.shape[-1] != last_dim:
                            s = s[:last_dim] if s.shape[-1] > last_dim else s
                        output_2d[start:end] = swiglu_2d[start:end] * s
            elif smooth_scales is not None:
                output_2d = output_2d * smooth_scales.float()

            # Per-row dynamic quantization
            y_max = torch.amax(torch.abs(output_2d), dim=-1, keepdim=True)
            y_max = torch.clamp(y_max, min=1e-10)
            dynamic_scale = float(int_scale) / y_max
            quantized_2d = torch.round(torch.clamp(output_2d * dynamic_scale, clip_min, clip_max)).to(torch.int8)
            
            if is_int4:
                quantized_2d = pack_int4_to_int8(quantized_2d)
                ori_shape = list(ori_shape)
                ori_shape[-1] = ori_shape[-1] // 2
            
            quantized = quantized_2d.reshape(ori_shape)
            quant_scales = dynamic_scale.squeeze(-1).reshape(prefix_shape)
            return quantized, quant_scales

        else:  # Static quantization (quant_mode == 0)
            # Apply group scales and offsets
            if smooth_scales is not None and boundaries is not None:
                smooth_f = smooth_scales.float()
                offsets_f = offsets.float() if offsets is not None else None
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    if start < end and i < smooth_f.shape[0]:
                        s = smooth_f[i]
                        if s.dim() >= 1 and s.shape[-1] != last_dim:
                            s = s[:last_dim] if s.shape[-1] > last_dim else s
                        output_2d[start:end] = swiglu_2d[start:end] * s
                        if offsets_f is not None and i < offsets_f.shape[0]:
                            o = offsets_f[i]
                            if o.dim() >= 1 and o.shape[-1] != last_dim:
                                o = o[:last_dim] if o.shape[-1] > last_dim else o
                            output_2d[start:end] = output_2d[start:end] + o
            elif smooth_scales is not None:
                output_2d = output_2d * smooth_scales.float()
                if offsets is not None:
                    output_2d = output_2d + offsets.float()

            quantized_2d = torch.round(output_2d).clamp(clip_min, clip_max).to(torch.int8)
            
            if is_int4:
                quantized_2d = pack_int4_to_int8(quantized_2d)
                ori_shape = list(ori_shape)
                ori_shape[-1] = ori_shape[-1] // 2
            
            quantized = quantized_2d.reshape(ori_shape)
            quant_scales = torch.zeros(prefix_shape, dtype=torch.float32)
            return quantized, quant_scales
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None, offsets: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                group_list_type: int = 0, dst_type = None) -> tuple:
        """
        Performs SwiGLU with quantization.

        Args:
            x (torch.Tensor): Target tensor. Must be >1D, last axis must be even and <= 8192.
                              dtype: float16, bfloat16, float32, format: ND.
                              For int4 quantization, last dim must be multiple of 4.
            smooth_scales (torch.Tensor, optional): Smooth quantization scale.
                                                    dtype: float32, format: ND. Shape: [G, N] or [G, ].
            offsets (torch.Tensor, optional): Quantization offset. Not used in dynamic quantization.
                                              dtype: float, format: ND. Shape must match smooth_scales.
            group_index (torch.Tensor, optional): Group index tensor (cumsum or count mode).
                                                  dtype: int32, format: ND. Shape: [G, ].
                                                  Must be non-decreasing, max <= product of non-last dims.
            activate_left (bool, optional): Whether to activate left in SwiGLU. Default: False.
            quant_mode (int, optional): Quantization type. 0: static, 1: dynamic. Default: 0.
            group_list_type (int, optional): Group index type. 0: cumsum, 1: count. Default: 0.
            dst_type: Output quantization type. Supports int8 and int4. None means int8. Default: None.

        Returns:
            tuple: (output tensor, quantization parameters) after SwiGLU quantization.
        """
        return torch_npu.npu_swiglu_quant(x, smooth_scales=smooth_scales, offsets=offsets,
                                          group_index=group_index, activate_left=activate_left,
                                          quant_mode=quant_mode, group_list_type=group_list_type,
                                          dst_type=dst_type)


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
            attrs = {}
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
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']

            x_shape = None
            for inp in inputs:
                if inp['name'] == 'x' and inp['type'] == 'tensor':
                    x_shape = inp.get('shape')
                    break

            if x_shape is not None:
                total_tokens = 1
                for s in x_shape[:-1]:
                    total_tokens *= s
                half_dim = x_shape[-1] // 2

                group_list_type = attrs.get('group_list_type', 0)
                quant_mode = attrs.get('quant_mode', 0)
                gi_shape = attrs.get('_group_index_shape')

                if 'group_index' in tensors and tensors['group_index'] is not None:
                    gi_shape_actual = tensors['group_index'].shape
                    num_groups = gi_shape_actual[0]
                    if group_list_type == 0:
                        group_size = total_tokens // num_groups
                        gi_vals = torch.tensor([(i + 1) * group_size for i in range(num_groups)], dtype=torch.int32)
                    else:
                        gi_vals = torch.tensor([total_tokens // num_groups] * num_groups, dtype=torch.int32)
                    tensors['group_index'] = gi_vals

                if 'smooth_scales' in tensors and tensors['smooth_scales'] is not None:
                    sshape = tensors['smooth_scales'].shape
                    if len(sshape) > 0:
                        num_groups = sshape[0]
                    else:
                        num_groups = 1
                    if len(sshape) == 2:
                        tensors['smooth_scales'] = torch.randn(num_groups, half_dim, dtype=torch.float32)
                    else:
                        tensors['smooth_scales'] = torch.randn(num_groups, dtype=torch.float32)

                if 'offsets' in tensors and tensors['offsets'] is not None:
                    if quant_mode == 1:
                        tensors['offsets'] = None
                    else:
                        ss_tensor = tensors.get('smooth_scales')
                        if ss_tensor is not None:
                            tensors['offsets'] = torch.randn(ss_tensor.shape, dtype=torch.float32)
                        else:
                            tensors['offsets'] = None

            dst_type_str = attrs.get('dst_type', 'int8')
            if dst_type_str == 'int8':
                dst_type_val = torch.int8
            elif dst_type_str == 'int4':
                dst_type_val = torch.quint4x2
            else:
                dst_type_val = None

            group = [
                tensors['x'],
                tensors.get('smooth_scales'),
                tensors.get('offsets'),
                tensors.get('group_index'),
                attrs.get('activate_left', False),
                attrs.get('quant_mode', 0),
                attrs.get('group_list_type', 0),
                dst_type_val
            ]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
