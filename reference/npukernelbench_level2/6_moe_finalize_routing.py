import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that finalizes routing for MoE (Mixture of Experts).
    torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode=0) -> Tensor
    Pure PyTorch implementation (replacing torch_npu.npu_moe_finalize_routing):

        def forward(self, expanded_permuted_rows, skip1, skip2, bias, scales,
                    expanded_src_to_dst_row, export_for_source_row, drop_pad_mode=0):
            # expanded_permuted_rows: [NUM_ROWS*K, H] (non-drop) or [E, C, H] (drop)
            # scales: [NUM_ROWS, K] expert weights
            # expanded_src_to_dst_row: [NUM_ROWS*K] index mapping
            # export_for_source_row: [NUM_ROWS, K] expert assignments

            if drop_pad_mode in [0, 2]:  # non-drop mode
                NUM_ROWS_K, H = expanded_permuted_rows.shape
                NUM_ROWS = scales.shape[0]
                K = scales.shape[1]

                # Gather rows according to expanded_src_to_dst_row
                gathered = expanded_permuted_rows[expanded_src_to_dst_row.long()]

                # Reshape depends on column vs row arrangement
                if drop_pad_mode == 0:  # column arrangement: flat idx = r + k*NUM_ROWS
                    gathered = gathered.view(K, NUM_ROWS, H).transpose(0, 1)
                else:  # drop_pad_mode == 2, row arrangement: flat idx = r*K + k
                    gathered = gathered.view(NUM_ROWS, K, H)

                # Apply scales: [NUM_ROWS, K, 1] * [NUM_ROWS, K, H]
                output = gathered * scales.unsqueeze(-1)

                # Sum over K dimension: [NUM_ROWS, H]
                output = output.sum(dim=1)

            elif drop_pad_mode in [1, 3]:  # drop mode
                E, C, H = expanded_permuted_rows.shape
                NUM_ROWS = scales.shape[0]
                K = scales.shape[1]

                # Use expanded_src_to_dst_row to compute (expert, slot) pairs
                output = torch.zeros(NUM_ROWS, H, device=expanded_permuted_rows.device,
                                     dtype=expanded_permuted_rows.dtype)
                for r in range(NUM_ROWS):
                    for k in range(K):
                        if drop_pad_mode == 1:  # column arrangement
                            flat_idx = expanded_src_to_dst_row[r + k * NUM_ROWS].long()
                        else:  # drop_pad_mode == 3, row arrangement
                            flat_idx = expanded_src_to_dst_row[r * K + k].long()
                        expert = flat_idx // C
                        slot = flat_idx % C
                        if 0 <= expert < E and 0 <= slot < C:
                            output[r] += expanded_permuted_rows[expert, slot] * scales[r, k]

            # Add bias if provided
            if bias is not None:
                # bias: [E, H], need to select based on expert assignment
                if export_for_source_row is not None:
                    row_bias = torch.zeros(NUM_ROWS, H, device=output.device, dtype=output.dtype)
                    for k in range(K):
                        expert_idx = export_for_source_row[:, k]
                        row_bias += bias[expert_idx] * scales[:, k:k+1]
                    output = output + row_bias

            # Add skip connections if provided
            if skip1 is not None:
                output = output + skip1
            if skip2 is not None:
                output = output + skip2

            return output
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expanded_permuted_rows: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor,
                bias: torch.Tensor, scales: torch.Tensor, expanded_src_to_dst_row: torch.Tensor,
                export_for_source_row: torch.Tensor, drop_pad_mode: int = 0) -> torch.Tensor:
        """
        Finalizes routing for MoE.

        Args:
            expanded_permuted_rows (torch.Tensor): Result processed by experts, must be 2D.
                                                   dtype: float16, bfloat16, float32, format: ND.
                                                   drop_pad_mode 0/2: shape (NUM_ROWS*K, H)
                                                   drop_pad_mode 1/3: shape (E, C, H)
            skip1 (torch.Tensor): Sum input param 1, can be None. Must be 2D, same dtype and shape as output.
            skip2 (torch.Tensor): Sum input param 2, can be None. Must be 2D, same dtype and shape as output.
                                  If skip1 is None, skip2 must also be None.
            bias (torch.Tensor): Expert bias, can be None. Must be 2D, same dtype as expanded_permuted_rows.
                                 Shape: (E, H).
            scales (torch.Tensor): Expert weights, can be None. Must be 2D, same dtype as expanded_permuted_rows.
                                   Shape: (NUM_ROWS, K).
            expanded_src_to_dst_row (torch.Tensor): Index of each expert's processing result. Must be 1D.
                                                    dtype: int32. Shape: (NUM_ROWS*K).
            export_for_source_row (torch.Tensor): Expert number for each row, can be None. Must be 2D.
                                                  dtype: int32. Shape: (NUM_ROWS, K). Range: [0, E-1].
            drop_pad_mode (int, optional): Drop mode and arrangement. Range: [0, 3]. Default: 0.
                                           0: non-drop mode, column arrangement
                                           1: drop mode, column arrangement
                                           2: non-drop mode, row arrangement
                                           3: drop mode, row arrangement

        Returns:
            torch.Tensor: Output tensor after finalizing MoE routing.
        """
        return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales,
                                                   expanded_src_to_dst_row, export_for_source_row,
                                                   drop_pad_mode=drop_pad_mode)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_moe_finalize_routing.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        expanded_permuted_rows_info = inputs[0]
        skip1_info = inputs[1]
        skip2_info = inputs[2]
        bias_info = inputs[3]
        scales_info = inputs[4]
        expanded_src_to_dst_row_info = inputs[5]
        export_for_source_row_info = inputs[6]
        drop_pad_mode_info = inputs[7]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[expanded_permuted_rows_info["dtype"]]
        
        expanded_permuted_rows = torch.randn(expanded_permuted_rows_info["shape"], dtype=dtype)
        
        if skip1_info.get("shape") is None:
            skip1 = None
            skip2 = None
        else:
            skip1 = torch.randn(skip1_info["shape"], dtype=dtype)
            skip2 = torch.randn(skip2_info["shape"], dtype=dtype)
        
        if bias_info.get("shape") is None:
            bias = None
        else:
            bias = torch.randn(bias_info["shape"], dtype=dtype)
        
        scales = torch.randn(scales_info["shape"], dtype=dtype)
        expanded_src_to_dst_row = torch.randint(0, expanded_permuted_rows_info["shape"][0], expanded_src_to_dst_row_info["shape"], dtype=torch.int32)
        export_for_source_row = torch.randint(0, 8, export_for_source_row_info["shape"], dtype=torch.int32)
        drop_pad_mode = drop_pad_mode_info["value"]
        input_groups.append([expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode])
    return input_groups


def get_init_inputs():
    return []
