import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs Fusion Attention computation using NPU accelerated npu_fusion_attention.
    Implements fused Transformer Attention Score computation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                head_num: int, input_layout: str, *, pse=None, padding_mask=None,
                atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647,
                next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
        """
        Performs fused attention computation on NPU.

        Args:
            query (Tensor): Query tensor, dtype float16/bfloat16/float32.
            key (Tensor): Key tensor, same dtype as query.
            value (Tensor): Value tensor, same dtype as query.
            head_num (int): Number of attention heads.
            input_layout (str): Data layout, supports BSH/SBH/BSND/BNSD/TND.
            pse (Tensor, optional): Position encoding.
            padding_mask (Tensor, optional): Not supported yet.
            atten_mask (Tensor, optional): Attention mask, dtype bool/uint8.
            scale (float): Scaling factor, default 1.
            keep_prob (float): Dropout keep probability, default 1.
            pre_tockens (int): Sparse computation parameter, default 2147483647.
            next_tockens (int): Sparse computation parameter, default 2147483647.
            inner_precise (int): Precision control, default 0.
            prefix (List[int], optional): Prefix sparse computation parameter.
            actual_seq_qlen (List[int], optional): Cumulative query sequence lengths for varlen.
            actual_seq_kvlen (List[int], optional): Cumulative key/value sequence lengths for varlen.
            sparse_mode (int): Sparse mode, default 0.
            gen_mask_parallel (bool): DSA parallel control, default True.
            sync (bool): DSA sync control, default False.
            softmax_layout (str): Softmax output layout for TND, default "".
            sink (Tensor, optional): Per-head bias, shape [head_num], dtype float32.

        Returns:
            tuple: (attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length)
        """
        import torch_npu
        return torch_npu.npu_fusion_attention(query, key, value, head_num, input_layout,
                                               pse=pse, padding_mask=padding_mask,
                                               atten_mask=atten_mask, scale=scale,
                                               keep_prob=keep_prob, pre_tockens=pre_tockens,
                                               next_tockens=next_tockens, inner_precise=inner_precise,
                                               prefix=prefix, actual_seq_qlen=actual_seq_qlen,
                                               actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode,
                                               gen_mask_parallel=gen_mask_parallel, sync=sync,
                                               softmax_layout=softmax_layout, sink=sink)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_fusion_attention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]
        head_num_info = inputs[3]
        input_layout_info = inputs[4]

        dtype = dtype_map[query_info["dtype"]]

        query = torch.randn(query_info["shape"], dtype=dtype)
        key = torch.randn(key_info["shape"], dtype=dtype)
        value = torch.randn(value_info["shape"], dtype=dtype)
        head_num = head_num_info["value"]
        input_layout = input_layout_info["value"]

        pse = None
        padding_mask = None
        atten_mask = None
        scale = 1.
        keep_prob = 1.
        pre_tockens = 2147483647
        next_tockens = 2147483647
        inner_precise = 0
        prefix = None
        actual_seq_qlen = None
        actual_seq_kvlen = None
        sparse_mode = 0
        gen_mask_parallel = True
        sync = False
        softmax_layout = ""
        sink = None

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "pse":
                pse = torch.randn(inp["shape"], dtype=dtype)
            elif name == "padding_mask":
                padding_mask = None
            elif name == "atten_mask":
                atten_mask_dtype_map = {
                    "bool": torch.bool,
                    "uint8": torch.uint8,
                }
                atten_mask_dtype = atten_mask_dtype_map.get(inp.get("dtype", "bool"), torch.bool)
                atten_mask = torch.ones(inp["shape"], dtype=atten_mask_dtype)
            elif name == "scale":
                scale = inp["value"]
            elif name == "keep_prob":
                keep_prob = inp["value"]
            elif name == "pre_tockens":
                pre_tockens = inp["value"]
            elif name == "next_tockens":
                next_tockens = inp["value"]
            elif name == "inner_precise":
                inner_precise = inp["value"]
            elif name == "prefix":
                prefix = inp["value"]
            elif name == "actual_seq_qlen":
                actual_seq_qlen = inp["value"]
            elif name == "actual_seq_kvlen":
                actual_seq_kvlen = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "gen_mask_parallel":
                gen_mask_parallel = inp["value"]
            elif name == "sync":
                sync = inp["value"]
            elif name == "softmax_layout":
                softmax_layout = inp["value"]
            elif name == "sink":
                sink = torch.randn(inp["shape"], dtype=torch.float32)

        input_groups.append([query, key, value, head_num, input_layout,
                             pse, padding_mask, atten_mask, scale, keep_prob,
                             pre_tockens, next_tockens, inner_precise, prefix,
                             actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                             gen_mask_parallel, sync, softmax_layout, sink])
    return input_groups


def get_init_inputs():
    return []
