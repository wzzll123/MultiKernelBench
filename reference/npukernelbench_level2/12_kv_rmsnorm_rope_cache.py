import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs KV RMSNorm and RoPE with cache operations.
    torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
    PyTorch native implementation of forward function
    def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                index: torch.Tensor, k_cache: torch.Tensor, ckv_cache: torch.Tensor,
                k_rope_scale: torch.Tensor = None, c_kv_scale: torch.Tensor = None,
                k_rope_offset: torch.Tensor = None, c_kv_offset: torch.Tensor = None,
                epsilon: float = 1e-5, cache_mode: str = 'Norm', is_output_kv: bool = False) -> tuple:

        B, N, S, hidden_size = kv.shape
        rms_size = gamma.shape[0]
        rope_size = hidden_size - rms_size
        orig_dtype = kv.dtype

        kv = kv.float()
        gamma = gamma.float()
        cos = cos.float()
        sin = sin.float()

        kv = rearrange(kv, 'b n s d -> b s n d')
        cos = rearrange(cos, 'b n s d -> b s n d')
        sin = rearrange(sin, 'b n s d -> b s n d')

        rms_in = kv[..., :rms_size]
        rope_in = kv[..., rms_size:]

        rms_mean = torch.mean(rms_in ** 2, dim=-1, keepdim=True)
        rms_normalized = rms_in / torch.sqrt(rms_mean + epsilon)
        v = gamma * rms_normalized

        k = rope_in.reshape(B, S, N, rope_size // 2, 2).transpose(-1, -2).reshape(B, S, N, rope_size)
        k1 = k[..., :rope_size // 2]
        k2 = k[..., rope_size // 2:]
        rotate_half_k = torch.cat((-k2, k1), dim=-1)

        if cos.shape[1] == 1 and S > 1:
            cos = cos.expand(B, S, N, rope_size)
        if sin.shape[1] == 1 and S > 1:
            sin = sin.expand(B, S, N, rope_size)

        k_embed = k * cos + rotate_half_k * sin

        v_out = rearrange(v, 'b s n d -> b n s d').to(orig_dtype)
        k_embed_out = rearrange(k_embed, 'b s n d -> b n s d').to(orig_dtype)

        v_for_cache = v.clone()
        k_for_cache = k_embed.clone()

        if c_kv_scale is not None:
            v_for_cache = v_for_cache * c_kv_scale.float()
        if c_kv_offset is not None:
            v_for_cache = v_for_cache + c_kv_offset.float()
        if c_kv_scale is not None:
            v_for_cache = torch.round(v_for_cache).clamp(-128, 127)

        if k_rope_scale is not None:
            k_for_cache = k_for_cache * k_rope_scale.float()
        if k_rope_offset is not None:
            k_for_cache = k_for_cache + k_rope_offset.float()
        if k_rope_scale is not None:
            k_for_cache = torch.round(k_for_cache).clamp(-128, 127)

        # Cache operations
        k_cache_out = k_cache.clone()
        ckv_cache_out = ckv_cache.clone()

        if cache_mode == 'Norm':
            if index.dim() == 2:
                for b in range(min(B, index.shape[0])):
                    for s in range(min(S, index.shape[1])):
                        idx = index[b, s].item()
                        if idx < 0:
                            continue
                        if idx < k_cache_out.shape[2]:
                            k_cache_out[b, :, idx, :] = k_for_cache[b, s, :, :].to(k_cache_out.dtype)
                            ckv_cache_out[b, :, idx, :] = v_for_cache[b, s, :, :].to(ckv_cache_out.dtype)
            elif index.dim() == 1:
                for i, idx_t in enumerate(index):
                    idx = idx_t.item()
                    if idx < 0:
                        continue
                    b_idx = i // S
                    s_idx = i % S
                    if b_idx < B and s_idx < S and idx < k_cache_out.shape[2]:
                        k_cache_out[b_idx, :, idx, :] = k_for_cache[b_idx, s_idx, :, :].to(k_cache_out.dtype)
                        ckv_cache_out[b_idx, :, idx, :] = v_for_cache[b_idx, s_idx, :, :].to(ckv_cache_out.dtype)

        elif cache_mode in ['PA', 'PA_BNSD']:
            if index.dim() == 1:
                block_size = k_cache_out.shape[1]
                k_flat = k_for_cache.reshape(B * S, N, -1)
                v_flat = v_for_cache.reshape(B * S, N, -1)
                cache_shape_k = k_cache_out.shape
                cache_shape_v = ckv_cache_out.shape
                k_cache_flat = k_cache_out.reshape(-1, N, cache_shape_k[-1])
                v_cache_flat = ckv_cache_out.reshape(-1, N, cache_shape_v[-1])

                for i in range(min(len(index), B * S)):
                    idx = index[i].item()
                    if idx < 0:
                        continue
                    if idx < k_cache_flat.shape[0]:
                        k_cache_flat[idx, :, :] = k_flat[i, :, :].to(k_cache_flat.dtype)
                        v_cache_flat[idx, :, :] = v_flat[i, :, :].to(v_cache_flat.dtype)

                k_cache_out = k_cache_flat.reshape(cache_shape_k)
                ckv_cache_out = v_cache_flat.reshape(cache_shape_v)

        elif cache_mode == 'PA_NZ':
            if index.dim() == 1:
                block_size = k_cache_out.shape[1]
                dk = k_cache_out.shape[-1]
                dv = ckv_cache_out.shape[-1]
                dk0 = 32 if k_cache_out.dtype == torch.int8 else 16
                dv0 = 32 if ckv_cache_out.dtype == torch.int8 else 16
                dk1 = dk // dk0
                dv1 = dv // dv0
                bn = k_cache_out.shape[0]
                num_head = k_cache_out.shape[2]

                k_cache_nz = k_cache_out.reshape(bn, num_head, dk1, block_size, dk0)
                v_cache_nz = ckv_cache_out.reshape(bn, num_head, dv1, block_size, dv0)

                k_flat = k_for_cache.reshape(B * S, N, -1)
                v_flat = v_for_cache.reshape(B * S, N, -1)

                for i in range(min(len(index), B * S)):
                    idx = index[i].item()
                    if idx < 0:
                        continue
                    bn_id = idx // block_size
                    block_offset = idx % block_size
                    if bn_id < bn:
                        for d in range(dk1):
                            k_cache_nz[bn_id, :, d, block_offset, :] = k_flat[i, :, d*dk0:(d+1)*dk0].to(k_cache_nz.dtype)
                        for d in range(dv1):
                            v_cache_nz[bn_id, :, d, block_offset, :] = v_flat[i, :, d*dv0:(d+1)*dv0].to(v_cache_nz.dtype)

                k_cache_out = k_cache_nz.reshape(k_cache_out.shape)
                ckv_cache_out = v_cache_nz.reshape(ckv_cache_out.shape)

        elif cache_mode == 'PA_BLK_BNSD':
            if index.dim() == 1:
                block_size = k_cache_out.shape[1]
                ceil_div_s = (S + block_size - 1) // block_size

                for batch in range(B):
                    for seq_id in range(ceil_div_s):
                        seq_start = seq_id * block_size
                        seq_end = S if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                        copy_len = seq_end - seq_start
                        idx_pos = batch * ceil_div_s + seq_id
                        if idx_pos >= len(index):
                            continue
                        idx_val = index[idx_pos].item()
                        if idx_val < 0:
                            continue
                        cache_b = idx_val // block_size
                        if cache_b < k_cache_out.shape[0]:
                            k_cache_out[cache_b, :copy_len, :, :] = k_for_cache[batch, seq_start:seq_end, :, :].to(k_cache_out.dtype)
                            ckv_cache_out[cache_b, :copy_len, :, :] = v_for_cache[batch, seq_start:seq_end, :, :].to(ckv_cache_out.dtype)

        elif cache_mode == 'PA_BLK_NZ':
            if index.dim() == 1:
                block_size = k_cache_out.shape[1]
                dk = k_cache_out.shape[-1]
                dv = ckv_cache_out.shape[-1]
                dk0 = 32 if k_cache_out.dtype == torch.int8 else 16
                dv0 = 32 if ckv_cache_out.dtype == torch.int8 else 16
                dk1 = dk // dk0
                dv1 = dv // dv0
                bn = k_cache_out.shape[0]
                num_head = k_cache_out.shape[2]
                ceil_div_s = (S + block_size - 1) // block_size

                k_cache_nz = k_cache_out.reshape(bn, num_head, dk1, block_size, dk0)
                v_cache_nz = ckv_cache_out.reshape(bn, num_head, dv1, block_size, dv0)

                for batch in range(B):
                    for seq_id in range(ceil_div_s):
                        seq_start = seq_id * block_size
                        seq_end = S if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
                        copy_len = seq_end - seq_start
                        idx_pos = batch * ceil_div_s + seq_id
                        if idx_pos >= len(index):
                            continue
                        idx_val = index[idx_pos].item()
                        if idx_val < 0:
                            continue
                        cache_b = idx_val // block_size
                        if cache_b < bn:
                            for n_idx in range(num_head):
                                for d in range(dk1):
                                    k_cache_nz[cache_b, n_idx, d, :copy_len, :] = k_for_cache[batch, seq_start:seq_end, n_idx, d*dk0:(d+1)*dk0].to(k_cache_nz.dtype)
                                for d in range(dv1):
                                    v_cache_nz[cache_b, n_idx, d, :copy_len, :] = v_for_cache[batch, seq_start:seq_end, n_idx, d*dv0:(d+1)*dv0].to(v_cache_nz.dtype)

                k_cache_out = k_cache_nz.reshape(k_cache_out.shape)
                ckv_cache_out = v_cache_nz.reshape(ckv_cache_out.shape)

        if is_output_kv:
            k_embed_ret = k_embed_out
            y_ret = v_out
        else:
            k_embed_ret = None
            y_ret = None

        return k_cache_out, ckv_cache_out, k_embed_ret, y_ret
    """
    def __init__(self):
        super(Model, self).__init__()

    def postprocess_output(self, output, inputs):
        """
        KV RMSNorm RoPE Cache 专用输出裁剪
        规则：Norm模式 或 is_output_kv=False → 只校验前两个输出
        """
        # 输入顺序与 forward 完全一致
        # kv, gamma, cos, sin, index, k_cache, ckv_cache,
        # k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset,
        # epsilon, cache_mode, is_output_kv
        if len(inputs) >= 14:
            cache_mode = inputs[12]
            is_output_kv = inputs[13]

            if cache_mode == 'Norm' or not is_output_kv:
                return output[:2]

        return output

    def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                index: torch.Tensor, k_cache: torch.Tensor, ckv_cache: torch.Tensor,
                k_rope_scale: torch.Tensor = None, c_kv_scale: torch.Tensor = None,
                k_rope_offset: torch.Tensor = None, c_kv_offset: torch.Tensor = None,
                epsilon: float = 1e-5, cache_mode: str = 'Norm', is_output_kv: bool = False) -> tuple:
        """
        Performs KV RMSNorm and RoPE with cache operations.

        Args:
            kv (torch.Tensor): Input feature tensor. Must be 4D [batch_size, 1, seq_len, hidden_size].
                               hidden_size = rms_size + rope_size.
                               dtype: bfloat16, float16, format: BNSD.
            gamma (torch.Tensor): RMS normalization scale parameter. Must be 1D [rms_size].
                                  dtype: bfloat16, float16, format: ND.
            cos (torch.Tensor): RoPE cosine component. Must be 4D [batch_size, 1, seq_len, rope_size].
                                dtype: bfloat16, float16, format: ND.
            sin (torch.Tensor): RoPE sine component. Must be 4D [batch_size, 1, seq_len, rope_size].
                                dtype: bfloat16, float16, format: ND.
            index (torch.Tensor): Cache index tensor for locating write positions in caches.
                                  dtype: int64, format: ND. Shape depends on cache_mode.
            k_cache (torch.Tensor): Storage for quantized/non-quantized key vectors.
                                    dtype: bfloat16, float16, int8, format: ND. Shape depends on cache_mode.
            ckv_cache (torch.Tensor): Storage for quantized/non-quantized compressed KV vectors.
                                      dtype: bfloat16, float16, int8, format: ND. Shape depends on cache_mode.
            k_rope_scale (torch.Tensor, optional): K RoPE quantization scale. Must be 1D [rope_size].
                                                   dtype: float32, format: ND. Required in quantization mode.
            c_kv_scale (torch.Tensor, optional): Compressed KV quantization scale. Must be 1D [rms_size].
                                                 dtype: float32, format: ND. Required in quantization mode.
            k_rope_offset (torch.Tensor, optional): K RoPE quantization offset. Must be 1D [rope_size].
                                                    dtype: float32, format: ND. Required in quantization mode.
            c_kv_offset (torch.Tensor, optional): Compressed KV quantization offset. Must be 1D [rms_size].
                                                  dtype: float32, format: ND. Required in quantization mode.
            epsilon (float, optional): Small value for RMS normalization to prevent division by zero.
                                       Default: 1e-5.
            cache_mode (str, optional): Cache mode. Options: 'Norm', 'PA', 'PA_BNSD', 'PA_NZ', 
                                        'PA_BLK_BNSD', 'PA_BLK_NZ'. Default: 'Norm'.
            is_output_kv (bool, optional): Whether to output processed k_embed_out and y_out.
                                           Default: False. Only effective in PA modes.

        Returns:
            tuple: (k_cache, ckv_cache, k_embed_out, y_out) tensors. Last two are None if is_output_kv=False.
        """
        return torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                                    k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                                    k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                                    epsilon=epsilon, cache_mode=cache_mode,
                                                    is_output_kv=is_output_kv)


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

            # Generate valid index values based on cache_mode and cache shapes
            index = tensors.get('index')
            cache_mode = attrs.get('cache_mode', 'Norm')
            k_cache = tensors.get('k_cache')

            if index is not None and k_cache is not None:
                device = index.device
                if cache_mode == 'Norm':
                    # Norm：全局唯一索引 mod 最大序列长度，避免重复
                    max_seq = k_cache.shape[2]
                    if index.dim() == 2:
                        B, S = index.shape
                        total = B * S
                        index = (torch.arange(total, dtype=torch.int64, device=device) % max_seq).reshape(B, S)
                    else:
                        total = index.numel()
                        index = (torch.arange(total, dtype=torch.int64, device=device) % max_seq).reshape(index.shape)

                elif cache_mode in ('PA', 'PA_BNSD', 'PA_NZ'):
                    # PA 系列：直接生成连续唯一索引
                    index = torch.arange(index.numel(), dtype=torch.int64, device=device)

                elif cache_mode in ('PA_BLK_BNSD', 'PA_BLK_NZ'):
                    # 分块 PA：索引 = 连续序号 × block_size
                    block_size = k_cache.shape[1]
                    length = index.numel()
                    index = torch.arange(length, dtype=torch.int64, device=device) * block_size

                tensors['index'] = index
            group = [
                tensors['kv'], tensors['gamma'], tensors['cos'], tensors['sin'],
                tensors['index'], tensors['k_cache'], tensors['ckv_cache'],
                tensors.get('k_rope_scale'), tensors.get('c_kv_scale'),
                tensors.get('k_rope_offset'), tensors.get('c_kv_offset'),
                attrs.get('epsilon', 1e-5), attrs.get('cache_mode', 'Norm'),
                attrs.get('is_output_kv', False)
            ]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
