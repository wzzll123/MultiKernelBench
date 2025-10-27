import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Cross-attention between encoder and decoder sequences.
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, q, kv):
        out, _ = self.mha(q, kv, kv)
        return out

batch_size = 16
len_q, len_kv = 64, 128
d_model = 512

def get_inputs():
    q = torch.randn(batch_size, len_q, d_model)
    kv = torch.randn(batch_size, len_kv, d_model)
    return [q, kv]

def get_init_inputs():
    return [d_model, 8]
