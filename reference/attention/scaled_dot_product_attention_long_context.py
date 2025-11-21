import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Implements standard scaled dot-product attention.
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        output = F.scaled_dot_product_attention(Q, K, V)
        return output

batch_size = 1
seq_len = 16384
d_model = 12288

def get_inputs():
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    return [Q, K, V]

def get_init_inputs():
    return []
