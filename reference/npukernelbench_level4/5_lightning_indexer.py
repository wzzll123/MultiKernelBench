import torch
import torch.nn as nn
import json
import os
import numpy as np

class Model(nn.Module):
    """
    Model that performs Lightning Indexer computation using NPU accelerated npu_lightning_indexer.
    Computes Top-k positions for each token based on index queries and keys.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *,
                actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                block_table=None, layout_query="BSND", layout_key="BSND",
                sparse_count=2048, sparse_mode=3, pre_tokens=None, next_tokens=None,
                return_value=False):
        """
        Performs lightning indexer computation on NPU.

        Args:
            query (Tensor): Index query tensor, shape [B, S1, N1, D] or [T1, N1, D], dtype bfloat16/float16.
            key (Tensor): Index key tensor, shape [B, S2, N2, D] or PA_BSND/TND, dtype bfloat16/float16.
            weights (Tensor): Weight tensor, shape [B, S1, N1] or [T, N1], dtype bfloat16/float16.
            actual_seq_lengths_query (Tensor, optional): Valid token count per batch for query, dtype int32.
            actual_seq_lengths_key (Tensor, optional): Valid token count per batch for key, dtype int32.
            block_table (Tensor, optional): PageAttention block mapping table, dtype int32.
            layout_query (str): Query data layout, supports BSND/TND, default "BSND".
            layout_key (str): Key data layout, supports PA_BSND/BSND/TND, default "BSND".
            sparse_count (int): Number of blocks to retain in topK, range [1, 2048], default 2048.
            sparse_mode (int): Sparse mode, supports 0/3, default 3.
            pre_tokens (int, optional): Forward token count for sparse computation, default 2^63-1.
            next_tokens (int, optional): Backward token count for sparse computation, default 2^63-1.
            return_value (bool): Whether to output sparse_values, default False.

        Returns:
            tuple: (sparse_indices, sparse_values) where sparse_indices is int32 and sparse_values is int32.
        """
        import torch_npu
        if pre_tokens is None:
            pre_tokens = (1 << 63) - 1
        if next_tokens is None:
            next_tokens = (1 << 63) - 1
        return torch_npu.npu_lightning_indexer(
            query, key, weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table, layout_query=layout_query, layout_key=layout_key,
            sparse_count=sparse_count, sparse_mode=sparse_mode,
            pre_tokens=pre_tokens, next_tokens=next_tokens,
            return_value=return_value)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_lightning_indexer.json")
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
        weights_info = inputs[2]

        dtype = dtype_map[query_info["dtype"]]

        query = torch.tensor(np.random.uniform(-10, 10, query_info["shape"])).to(dtype)
        key = torch.tensor(np.random.uniform(-10, 10, key_info["shape"])).to(dtype)
        weights = torch.tensor(np.random.uniform(-1, 1, weights_info["shape"])).to(dtype)

        actual_seq_lengths_query = None
        actual_seq_lengths_key = None
        block_table = None
        layout_query = "BSND"
        layout_key = "BSND"
        sparse_count = 2048
        sparse_mode = 3
        pre_tokens = None
        next_tokens = None
        return_value = False

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "actual_seq_lengths_query":
                actual_seq_lengths_query = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "actual_seq_lengths_key":
                actual_seq_lengths_key = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "block_table":
                block_table = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "layout_query":
                layout_query = inp["value"]
            elif name == "layout_key":
                layout_key = inp["value"]
            elif name == "sparse_count":
                sparse_count = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "pre_tokens":
                pre_tokens = inp["value"]
            elif name == "next_tokens":
                next_tokens = inp["value"]
            elif name == "return_value":
                return_value = inp["value"]

        input_groups.append([query, key, weights,
                             actual_seq_lengths_query, actual_seq_lengths_key,
                             block_table, layout_query, layout_key,
                             sparse_count, sparse_mode, pre_tokens, next_tokens,
                             return_value])
    return input_groups


def get_init_inputs():
    return []
