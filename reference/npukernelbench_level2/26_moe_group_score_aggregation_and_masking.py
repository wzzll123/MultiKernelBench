import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    MoE Group-based Score Aggregation and Masking Module.

    Implements group-based routing for Mixture of Experts:
    1. Reshape expert scores into groups
    2. Compute top-2 scores per group and sum them for group quality
    3. Select top-k groups based on aggregated scores
    4. Mask out experts from non-selected groups
    """
    def __init__(self):
        super(Model, self).__init__()
        self.num_experts = 256
        self.n_group = 8
        self.topk_group = 4

    def forward(self, scores: torch.Tensor):
        """
        Group-based score aggregation and masking for MoE routing.

        Args:
            scores: Expert scores after sigmoid activation, shape (num_tokens, 256)

        Returns:
            masked_scores: Scores with non-selected groups masked, shape (num_tokens, 256)
            group_mask: Binary mask of selected groups, shape (num_tokens, 8)
        """
        experts_per_group = self.num_experts // self.n_group
        num_tokens = scores.size(0)
        group_scores_reshaped = scores.view(num_tokens, self.n_group, experts_per_group)
        top2_per_group = torch.topk(group_scores_reshaped, k=2, dim=-1)[0]
        group_scores = top2_per_group.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, self.n_group, experts_per_group).reshape(num_tokens, self.num_experts)
        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        return masked_scores, group_mask


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
                        if name == 'scores':
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
                group.append(tensors[inp['name']])
            input_groups.append(group)
            idx += 1
    return input_groups


def get_init_inputs():
    return []
