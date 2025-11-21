import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Implements standard multi-head attention (MHA).
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out

batch_size = 16
seq_len = 256
d_model = 512
num_heads = 8

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads]
