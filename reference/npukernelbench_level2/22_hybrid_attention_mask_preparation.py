import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hybrid causal mask preparation system that creates separate attention masks
    for full attention and sliding window attention layers.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        batch_size_scalar: int,
        seq_length_scalar: int,
        past_key_values_length_scalar: int,
        num_attention_heads: int = 64,
        swa_num_attention_heads: int = 64,
        sliding_window: int = 128,
    ) -> tuple:
        """
        Creates hybrid attention masks for full and sliding window attention.

        Args:
            batch_size_scalar (int): Batch size.
            seq_length_scalar (int): Current sequence length.
            past_key_values_length_scalar (int): Length of cached keys/values.
            num_attention_heads (int, optional): Number of heads for full attention. Default: 64.
            swa_num_attention_heads (int, optional): Number of heads for sliding window attention. Default: 64.
            sliding_window (int, optional): Window size for sliding window attention. Default: 128.

        Returns:
            tuple: (full_attention_mask, sliding_window_attention_mask)
                - full_attention_mask (torch.Tensor): Full causal attention mask
                                                      with shape [batch_size, num_attention_heads, seq_length, total_length].
                - sliding_window_attention_mask (torch.Tensor): Sliding window causal attention mask
                                                                with shape [batch_size, swa_num_attention_heads, seq_length, total_length].
        """
        batch_size = int(batch_size_scalar)
        seq_length = int(seq_length_scalar)
        past_key_values_length = int(past_key_values_length_scalar)

        target_length = seq_length
        source_length = seq_length + past_key_values_length

        full_mask = torch.ones(
            (target_length, source_length),
            dtype=torch.bool
        )

        target_indices = torch.arange(target_length)[:, None]
        source_indices = torch.arange(source_length)[None, :]
        causal_cond = target_indices >= (source_indices - past_key_values_length)
        full_mask = full_mask.masked_fill(causal_cond, False)

        full_attention_mask = full_mask[None, None, :, :].expand(
            batch_size, num_attention_heads, target_length, source_length
        ).contiguous()

        swa_mask = torch.zeros(
            (target_length, source_length),
            dtype=torch.bool
        )

        window_cond = (source_indices - past_key_values_length) >= (target_indices - sliding_window)

        valid_positions = causal_cond & window_cond
        swa_mask = swa_mask.masked_fill(valid_positions, False)

        sliding_window_attention_mask = swa_mask[None, None, :, :].expand(
            batch_size, swa_num_attention_heads, target_length, source_length
        ).contiguous()

        return full_attention_mask, sliding_window_attention_mask


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
