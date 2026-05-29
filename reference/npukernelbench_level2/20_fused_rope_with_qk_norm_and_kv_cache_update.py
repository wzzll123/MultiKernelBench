import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused operation combining per-head RMS normalization on Q and K, RoPE (Rotary Position Embedding) application,
    and KV cache update. This eliminates 5+ separate kernel launches by keeping intermediate results in registers/shared memory.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ) -> tuple:
        """
        Applies fused RMS norm + RoPE + KV cache update.

        Args:
            query (torch.Tensor): Query tensor with shape [batch_size, num_attention_heads, seq_len, head_dim].
                                  Supports bfloat16.
            key (torch.Tensor): Key tensor with shape [batch_size, num_key_value_heads, seq_len, head_dim].
                                Supports bfloat16.
            value (torch.Tensor): Value tensor with shape [batch_size, num_key_value_heads, seq_len, head_dim].
                                  Supports bfloat16.
            position_ids (torch.Tensor): Position indices for RoPE with shape [batch_size, seq_len]. Dtype is int64.
            key_cache (torch.Tensor): Key cache tensor with shape [batch_size, num_key_value_heads, max_position_embeddings, head_dim].
                                      Supports bfloat16.
            value_cache (torch.Tensor): Value cache tensor with shape [batch_size, num_key_value_heads, max_position_embeddings, head_dim].
                                        Supports bfloat16.
            cache_position (torch.Tensor): Positions in cache to update with shape [seq_len]. Dtype is int64.
            q_norm_weight (torch.Tensor): RMS norm weight for queries with shape [head_dim]. Supports bfloat16.
            k_norm_weight (torch.Tensor): RMS norm weight for keys with shape [head_dim]. Supports bfloat16.
            inv_freq (torch.Tensor): Inverse frequencies for RoPE with shape [half_head_dim]. Supports float32.
            rms_norm_eps (float): Epsilon for RMS normalization.

        Returns:
            tuple: (query_rotated, key_rotated, key_cache_out, value_cache_out)
                - query_rotated (torch.Tensor): Query with RMS norm and RoPE applied.
                - key_rotated (torch.Tensor): Key with RMS norm and RoPE applied.
                - key_cache_out (torch.Tensor): Updated key cache.
                - value_cache_out (torch.Tensor): Updated value cache.
        """
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        num_kv_heads = key.shape[1]

        def rms_norm(x, weight, eps):
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(-1, keepdim=True)
            x_normed = x_fp32 * torch.rsqrt(variance + eps)
            return (weight.to(torch.float32) * x_normed).to(x.dtype)

        query_norm = rms_norm(query, q_norm_weight, rms_norm_eps)
        key_norm = rms_norm(key, k_norm_weight, rms_norm_eps)

        inv_freq_expanded = inv_freq[None, None, :].expand(batch_size, seq_len, -1)
        position_ids_expanded = position_ids[:, :, None].float()
        freqs = position_ids_expanded * inv_freq_expanded
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(query.dtype)
        sin = emb.sin().to(query.dtype)

        def rotate_half(x):
            x1 = x[..., :head_dim // 2]
            x2 = x[..., head_dim // 2:]
            return torch.cat([-x2, x1], dim=-1)

        def apply_rope(x, cos, sin):
            cos_expanded = cos.unsqueeze(1)
            sin_expanded = sin.unsqueeze(1)
            return (x * cos_expanded) + (rotate_half(x) * sin_expanded)

        query_rotated = apply_rope(query_norm, cos, sin)
        key_rotated = apply_rope(key_norm, cos, sin)

        key_cache_out = key_cache.clone()
        value_cache_out = value_cache.clone()

        for i in range(seq_len):
            pos = cache_position[i].item()
            key_cache_out[:, :, pos, :] = key_rotated[:, :, i, :]
            value_cache_out[:, :, pos, :] = value[:, :, i, :]

        return query_rotated, key_rotated, key_cache_out, value_cache_out


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        idx = 0
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
                        if name == 'query':
                            if idx % 2 == 0:
                                mu = float(torch.empty(1).uniform_(-100, 100).item())
                                sigma = float(torch.empty(1).uniform_(1, 25).item())
                                tensors[name] = torch.normal(mu, sigma, shape, dtype=dtype) + torch.ones(shape, dtype=dtype)
                            else:
                                tensors[name] = torch.empty(shape, dtype=dtype).uniform_(-5, 5) + torch.ones(shape, dtype=dtype)
                        else:
                            tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                name = inp['name']
                val = tensors[name]
                if name == 'cache_position' and val is not None:
                    max_pos = tensors['value_cache'].shape[2]
                    val = torch.randint(0, max_pos, val.shape, dtype=val.dtype)
                group.append(val)
            input_groups.append(group)
            idx += 1
    return input_groups


def get_init_inputs():
    return []
