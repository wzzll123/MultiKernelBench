import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Computes 3D rotary position embeddings for multimodal inputs by generating position indices
    from grid_thw (temporal, height, width) parameters, performing bilinear interpolation for
    variable resolution support, and creating interleaved MRoPE embeddings.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grid_thw: torch.Tensor,
        pos_embed_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        spatial_merge_size: int = 2,
        num_grid_per_side: int = 35,
        mrope_section: list = None,
    ) -> tuple:
        """
        Computes multimodal RoPE position embeddings with grid-based indexing.

        Args:
            grid_thw (torch.Tensor): Grid dimensions [temporal_frames, height_patches, width_patches]
                                     for each input with shape [num_images, 3]. Dtype is int64.
            pos_embed_weight (torch.Tensor): Learned position embedding weights
                                             with shape [num_position_embeddings, hidden_size]. Supports float32.
            inv_freq (torch.Tensor): Precomputed inverse frequencies for rotary embeddings
                                     with shape [quarter_head_dim]. Supports float32.
            spatial_merge_size (int, optional): Size for spatial merging. Default: 2.
            num_grid_per_side (int, optional): Number of grid cells per side. Default: 35.
            mrope_section (list, optional): MRoPE section sizes. Default: [24, 20, 20].

        Returns:
            tuple: (patch_pos_embeds, cos_embeddings, sin_embeddings)
                - patch_pos_embeds (torch.Tensor): Interpolated position embeddings
                                                   with shape [total_tokens, hidden_size].
                - cos_embeddings (torch.Tensor): Cosine rotary embeddings
                                                 with shape [total_tokens, head_dim].
                - sin_embeddings (torch.Tensor): Sine rotary embeddings
                                                 with shape [total_tokens, head_dim].
        """
        if mrope_section is None:
            mrope_section = [24, 20, 20]

        dtype = pos_embed_weight.dtype

        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h.item(), device=grid_thw.device)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w.item(), device=grid_thw.device)

            h_idxs_floor = h_idxs.long()
            w_idxs_floor = w_idxs.long()
            h_idxs_ceil = (h_idxs_floor + 1).clamp(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs_floor + 1).clamp(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor.float()
            dw = w_idxs - w_idxs_floor.float()

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]

            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=grid_thw.device)
        weight_tensor = torch.tensor(weight_list, dtype=dtype, device=grid_thw.device)

        if dtype == torch.bfloat16:
            pos_embeds = pos_embed_weight[idx_tensor].float() * weight_tensor[:, :, None].float()
            patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
            patch_pos_embeds = patch_pos_embeds.to(dtype)
        else:
            pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
            patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h.item() * w.item() for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t.item(), 1)
            pos_embed = (
                pos_embed.view(t.item(), h.item() // merge_size, merge_size,
                              w.item() // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)

        max_hw = int(grid_thw[:, 1:].max().item())

        seq = torch.arange(max_hw, dtype=torch.float32, device=grid_thw.device)
        freqs = torch.outer(seq, inv_freq)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=grid_thw.device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height.item() // merge_size, width.item() // merge_size

            block_rows = torch.arange(merged_h, device=grid_thw.device)
            block_cols = torch.arange(merged_w, device=grid_thw.device)
            intra_row = torch.arange(merge_size, device=grid_thw.device)
            intra_col = torch.arange(merge_size, device=grid_thw.device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames.item(), 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freqs[pos_ids]
        embeddings = embeddings.flatten(1)

        freqs_3d = embeddings.unsqueeze(0).expand(3, -1, -1).clone()

        freqs_t = freqs_3d[0]
        for dim, offset_dim in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset_dim, length, 3)
            freqs_t[..., idx] = freqs_3d[dim, ..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos_embeddings = emb.cos()
        sin_embeddings = emb.sin()

        return patch_pos_embeds, cos_embeddings, sin_embeddings


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
                name = inp['name']
                val = tensors[name]
                if name == 'grid_thw' and val is not None:
                    merge_size = tensors.get('spatial_merge_size', 2)
                    val = (torch.randint(1, 8, val.shape, dtype=val.dtype) * merge_size)
                group.append(val)
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
