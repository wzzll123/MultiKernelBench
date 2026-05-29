import json
import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RWKV Time Decay Exponential with Numerical Stabilization Module.

    Computes time decay with exponential stabilization to prevent
    numerical overflow/underflow in recurrent state updates.

    Key computations:
    1. time_decay_exp = -exp(time_decay)
    2. For each timestep: max_state = max(max_state + time_decay_exp, current_key)
    3. Exponential normalization: e1 = exp(old_max - new_max), e2 = exp(key - new_max)
    """
    def __init__(self):
        super(Model, self).__init__()
        self.attention_hidden_size = 2048

    def forward(
        self,
        time_decay: torch.Tensor,
        key: torch.Tensor,
        time_first: torch.Tensor,
        value: torch.Tensor,
        max_state: torch.Tensor,
        num_state: torch.Tensor,
        den_state: torch.Tensor,
    ):
        """
        RWKV Time Decay Exponential with Numerical Stabilization.

        Args:
            time_decay: Decay rates [attention_hidden_size]
            key: Key tensor [batch_size, seq_len, attention_hidden_size]
            time_first: First time weights [attention_hidden_size]
            value: Value tensor [batch_size, seq_len, attention_hidden_size]
            max_state: Maximum state [batch_size, attention_hidden_size]
            num_state: Numerator state [batch_size, attention_hidden_size]
            den_state: Denominator state [batch_size, attention_hidden_size]

        Returns:
            output: Output tensor [batch_size, seq_len, attention_hidden_size]
            max_state: Updated maximum state
            num_state: Updated numerator state
            den_state: Updated denominator state
        """
        batch_size, seq_len, hidden_size = key.size()

        max_state = max_state.clone().float()
        num_state = num_state.clone().float()
        den_state = den_state.clone().float()

        time_decay_exp = -torch.exp(time_decay.float())

        output = torch.zeros_like(key, dtype=torch.float32)

        for t in range(seq_len):
            current_key = key[:, t].float()
            current_value = value[:, t].float()

            max_for_output = torch.maximum(
                max_state, current_key + time_first
            )

            e1_output = torch.exp(max_state - max_for_output)
            e2_output = torch.exp(current_key + time_first - max_for_output)

            numerator = e1_output * num_state + e2_output * current_value
            denominator = e1_output * den_state + e2_output
            output[:, t] = numerator / denominator

            max_for_state = torch.maximum(
                max_state + time_decay_exp, current_key
            )

            e1_state = torch.exp(max_state + time_decay_exp - max_for_state)
            e2_state = torch.exp(current_key - max_for_state)

            num_state = e1_state * num_state + e2_state * current_value
            den_state = e1_state * den_state + e2_state
            max_state = max_for_state

        return output, max_state, num_state, den_state


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
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
