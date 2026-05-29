import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that performs advance step for FlashAttention in vLLM.
    torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size) -> ()
    Pure PyTorch implementation (conceptual, as this is an in-place vLLM operation):

    def forward(self, input_tokens, sampled_token_ids, input_positions, seq_lens,
                slot_mapping, block_tables, num_seqs, num_queries, block_size):
        # This is a vLLM-specific scheduling operation that updates metadata
        # for FlashAttention. It's highly hardware-specific.

        # Update input_tokens with sampled_token_ids
        # Non-speculative: sampled_token_ids is [num_queries, 1]
        # input_tokens is [num_seqs,] or [num_seqs, 1+spec_num]
        if sampled_token_ids.dim() == 2:
            input_tokens[:num_queries] = sampled_token_ids.squeeze(-1)
        else:
            input_tokens[:num_queries] = sampled_token_ids

        # Update sequence lengths
        seq_lens[:num_queries] += 1

        # Update input positions
        input_positions[:num_queries] = seq_lens[:num_queries] - 1

        # Update slot_mapping (simplified, actual logic depends on block_tables)
        for i in range(num_queries):
            seq_len = int(seq_lens[i].item())
            block_idx = (seq_len - 1) // block_size
            block_offset = (seq_len - 1) % block_size
            if block_idx < block_tables.size(1):
                physical_block = block_tables[i, block_idx]
                slot_mapping[i] = physical_block * block_size + block_offset

        return None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor, 
                input_positions: torch.Tensor, seq_lens: torch.Tensor, 
                slot_mapping: torch.Tensor, block_tables: torch.Tensor,
                num_seqs: int, num_queries: int, block_size: int) -> None:
        """
        Performs advance step for FlashAttention in vLLM model.

        Args:
            input_tokens (torch.Tensor): Input/output tensor for updating token values in vLLM.
                                         dtype: int64. Non-speculative: [num_seqs,], Speculative: [num_seqs, 1 + spec_num].
            sampled_token_ids (torch.Tensor): Input tensor storing token_id. dtype: int64.
                                              Non-speculative: [num_queries, 1], Speculative: [num_seqs, 1 + spec_num].
            input_positions (torch.Tensor): Input/output tensor recording token index. dtype: int64.
                                            Non-speculative: [num_queries, 1], Speculative: [num_seqs, 1 + spec_num].
            seq_lens (torch.Tensor): Input/output tensor recording seq length under different block_idx. dtype: int64.
                                     Non-speculative: [num_queries, 1], Speculative: [num_seqs, 1 + spec_num].
            slot_mapping (torch.Tensor): Input/output tensor mapping token position to physical position. dtype: int64.
                                         Non-speculative: [num_queries, 1], Speculative: [num_seqs, 1 + spec_num].
            block_tables (torch.Tensor): Input/output tensor recording block size under different block_idx. dtype: int64.
                                         Shape: [num_seqs, max_blocks_per_seq].
            num_seqs (int): Number of input sequences. Must be > 0.
            num_queries (int): Number of input queries. Must be > 0.
            block_size (int): Size of each block. Must be > 0.

        Returns:
            None: This operation is in-place.
        """
        return torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions, 
                                                     seq_lens, slot_mapping, block_tables, 
                                                     num_seqs, num_queries, block_size)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_advance_step_flashattn.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tokens_info = inputs[0]
        sampled_token_ids_info = inputs[1]
        input_positions_info = inputs[2]
        seq_lens_info = inputs[3]
        slot_mapping_info = inputs[4]
        block_tables_info = inputs[5]
        num_seqs_info = inputs[6]
        num_queries_info = inputs[7]
        block_size_info = inputs[8]
        
        input_tokens = torch.randint(0, 1000, input_tokens_info["shape"], dtype=torch.int64)
        sampled_token_ids = torch.randint(0, 1000, sampled_token_ids_info["shape"], dtype=torch.int64)
        input_positions = torch.randint(0, 1000, input_positions_info["shape"], dtype=torch.int64)
        seq_lens = torch.randint(1, 100, seq_lens_info["shape"], dtype=torch.int64)
        slot_mapping = torch.randint(0, 10000, slot_mapping_info["shape"], dtype=torch.int64)
        block_tables = torch.randint(0, 100, block_tables_info["shape"], dtype=torch.int64)
        num_seqs = num_seqs_info["value"]
        num_queries = num_queries_info["value"]
        block_size = block_size_info["value"]
        input_groups.append([input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size])
    return input_groups


def get_init_inputs():
    return []
