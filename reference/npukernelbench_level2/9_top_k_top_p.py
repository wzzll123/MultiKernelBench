import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs top-k and top-p filtering.
    torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor

    PyTorch native implementation of forward function
    def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        ori_dtype = logits.dtype
        p = p.to(torch.float32)

        logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)
        kth_idx = logits_sort.size(1) - k.to(torch.long)
        kth_value = logits_sort.gather(1, kth_idx.unsqueeze(dim=1))
        top_k_mask = logits_sort < kth_value
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

        softmax_res = logits_sort.to(torch.float32).softmax(dim=-1)
        cumsum_res = softmax_res.cumsum(dim=-1)
        top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

        logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)

        return logits.to(ori_dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Performs top-k and top-p filtering on logits.

        Args:
            logits (torch.Tensor): Data to process. Must be 2D.
                                   dtype: float32, float16, bfloat16, format: ND.
                                   Supports non-contiguous tensors.
            p (torch.Tensor): Top-p threshold tensor. Range: [0, 1].
                              dtype: float32, float16, bfloat16 (must match logits).
                              Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.
            k (torch.Tensor): Top-k threshold tensor. Range: [1, 1024], max <= logits.size(1).
                              dtype: int32. Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.

        Returns:
            torch.Tensor: Filtered logits tensor.
        """
        return torch_npu.npu_top_k_top_p(logits, p, k)


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
                        if name == 'logits':
                            tensors[name] = ((torch.randn(shape) * 2.0).exp() - 2.0).clamp(max=torch.finfo(dtype).max).to(dtype)
                        else:
                            tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Clamp k and p to valid ranges for npu_top_k_top_p
            if 'logits' in tensors and tensors['logits'] is not None:
                N = tensors['logits'].shape[-1]
                if 'k' in tensors and tensors['k'] is not None:
                    tensors['k'] = tensors['k'].clamp(min=1, max=min(1024, N))
                if 'p' in tensors and tensors['p'] is not None:
                    tensors['p'] = 0.5 + torch.rand(tensors['p'].shape, dtype=tensors['p'].dtype) * 0.5

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
