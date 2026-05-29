import torch
import torch.nn as nn
import json
import os
import numpy as np
import random

class Model(nn.Module):
    """
    Model that performs Sparse Flash Attention computation using NPU accelerated npu_sparse_flash_attention.
    Efficiently computes attention for large sequence length inference with sparse key/value selection.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                sparse_indices: torch.Tensor, scale_value: float, *,
                block_table=None, actual_seq_lengths_query=None,
                actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                sparse_block_size=1, layout_query='BSND', layout_kv='BSND',
                sparse_mode=3, pre_tokens=None, next_tokens=None,
                attention_mode=0, return_softmax_lse=False):
        """
        Performs sparse flash attention on NPU.

        Args:
            query (Tensor): Query tensor, shape [B, S1, N1, D] or [T1, N1, D], dtype bfloat16/float16.
            key (Tensor): Key tensor, shape [B, S2, N2, D] or PA_BSND/TND, dtype bfloat16/float16.
            value (Tensor): Value tensor, same shape as key, dtype bfloat16/float16.
            sparse_indices (Tensor): Sparse KV cache indices, shape [B, S1, N2, sparse_size] or [T1, N2, sparse_size], dtype int32.
            scale_value (double): Scaling factor for QK^T.
            block_table (Tensor, optional): PageAttention block mapping table, dtype int32.
            actual_seq_lengths_query (Tensor, optional): Valid token count per batch for query, dtype int32.
            actual_seq_lengths_kv (Tensor, optional): Valid token count per batch for key/value, dtype int32.
            query_rope (Tensor, optional): Query rope info for MLA, dtype bfloat16/float16.
            key_rope (Tensor, optional): Key rope info for MLA, dtype bfloat16/float16.
            sparse_block_size (int): Block size for sparse stage, range [1, 128], power of 2, default 1.
            layout_query (str): Query layout, supports BSND/TND, default "BSND".
            layout_kv (str): Key/value layout, supports TND/BSND/PA_BSND, default "BSND".
            sparse_mode (int): Sparse mode, 0=all compute, 3=rightDownCausal, default 3.
            pre_tokens (int, optional): Forward token count, default 2^63-1.
            next_tokens (int, optional): Backward token count, default 2^63-1.
            attention_mode (int): Attention mode, only supports 2 (MLA-absorb), default 0.
            return_softmax_lse (bool): Whether to return softmax_max and softmax_sum, default False.

        Returns:
            tuple: (attention_out, softmax_max, softmax_sum)
        """
        import torch_npu
        if pre_tokens is None:
            pre_tokens = (1 << 63) - 1
        if next_tokens is None:
            next_tokens = (1 << 63) - 1
        return torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope, key_rope=key_rope,
            sparse_block_size=sparse_block_size,
            layout_query=layout_query, layout_kv=layout_kv,
            sparse_mode=sparse_mode, pre_tokens=pre_tokens, next_tokens=next_tokens,
            attention_mode=attention_mode, return_softmax_lse=return_softmax_lse)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_sparse_flash_attention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]
        sparse_indices_info = inputs[3]
        scale_value_info = inputs[4]

        dtype = dtype_map[query_info["dtype"]]

        query = torch.tensor(np.random.uniform(-10, 10, query_info["shape"])).to(dtype)
        key = torch.tensor(np.random.uniform(-5, 10, key_info["shape"])).to(dtype)
        value = key.clone()
        sparse_indices = torch.randint(0, 8192, sparse_indices_info["shape"], dtype=torch.int32)
        scale_value = scale_value_info["value"]

        block_table = None
        actual_seq_lengths_query = None
        actual_seq_lengths_kv = None
        query_rope = None
        key_rope = None
        sparse_block_size = 1
        layout_query = 'BSND'
        layout_kv = 'BSND'
        sparse_mode = 3
        pre_tokens = None
        next_tokens = None
        attention_mode = 0
        return_softmax_lse = False

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "block_table":
                block_table = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "actual_seq_lengths_query":
                actual_seq_lengths_query = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "actual_seq_lengths_kv":
                actual_seq_lengths_kv = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "query_rope":
                query_rope = torch.tensor(np.random.uniform(-10, 10, inp["shape"])).to(dtype)
            elif name == "key_rope":
                key_rope = torch.tensor(np.random.uniform(-10, 10, inp["shape"])).to(dtype)
            elif name == "sparse_block_size":
                sparse_block_size = inp["value"]
            elif name == "layout_query":
                layout_query = inp["value"]
            elif name == "layout_kv":
                layout_kv = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "pre_tokens":
                pre_tokens = inp["value"]
            elif name == "next_tokens":
                next_tokens = inp["value"]
            elif name == "attention_mode":
                attention_mode = inp["value"]
            elif name == "return_softmax_lse":
                return_softmax_lse = inp["value"]

        input_groups.append([query, key, value, sparse_indices, scale_value,
                             block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                             query_rope, key_rope, sparse_block_size,
                             layout_query, layout_kv, sparse_mode,
                             pre_tokens, next_tokens, attention_mode, return_softmax_lse])
    return input_groups


def get_init_inputs():
    return []
