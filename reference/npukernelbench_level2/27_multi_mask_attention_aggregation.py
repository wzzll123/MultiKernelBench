import json
import os
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Multi-mask attention map aggregation with softmax normalization.
    
    This operator performs:
    1. Softmax normalization on attention scores
    2. Weighted masking for each class using ref_target_masks
    3. Sum normalization: weighted sum divided by mask sum
    4. Dimension permutation: (B, H, seq_len) -> (B, seq_len, H)
    5. Aggregation across heads: mean or max pooling
    
    Input:
        - attn: Attention scores of shape (B, H, x_seq_len, ref_seq_len)
        - ref_target_masks: Class masks of shape (num_classes, ref_seq_len)
        - mode: Aggregation mode, 'mean' or 'max'
    
    Output:
        - Attention maps tensor of shape (num_classes, B, x_seq_len)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        attn: torch.Tensor,
        ref_target_masks: torch.Tensor,
        mode: str = "mean",
    ):
        x_ref_attn_map_source = attn.softmax(-1)

        x_ref_attn_maps = []

        ref_target_masks = ref_target_masks.to(attn.dtype)
        x_ref_attn_map_source = x_ref_attn_map_source.to(attn.dtype)

        for class_idx, ref_target_mask in enumerate(ref_target_masks):
            ref_target_mask = ref_target_mask[None, None, None, ...]

            x_ref_attnmap = x_ref_attn_map_source * ref_target_mask

            x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum()

            x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)

            if mode == "mean":
                x_ref_attnmap = x_ref_attnmap.mean(-1)
            elif mode == "max":
                x_ref_attnmap = x_ref_attnmap.max(-1)[0]

            x_ref_attn_maps.append(x_ref_attnmap)

        return torch.stack(x_ref_attn_maps, dim=0)


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
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        if name == 'ref_target_masks':
                            tensors[name] = (torch.rand(shape) > 0.5).to(dtype)
                        else:
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
