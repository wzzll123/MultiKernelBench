import torch
import torch.nn as nn
import math
import time

class Model(nn.Module):
    """
    Multi-Query Attention (MQA)
    - Multiple query heads
    - Single shared key/value head
    """
    def __init__(self, d_model=4096, num_heads=32):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model // num_heads)  # shared across heads
        self.v_proj = nn.Linear(d_model, d_model // num_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d_h = D // H

        Q = self.q_proj(x).view(B, N, H, d_h)
        K = self.k_proj(x).unsqueeze(2)  # shared K/V
        V = self.v_proj(x).unsqueeze(2)

        attn = torch.einsum("bnhd,bkhd->bh nk", Q, K) / math.sqrt(d_h)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnk,bkhd->bnhd", attn, V).reshape(B, N, D)
        return self.out_proj(out)

batch_size, seq_len, d_model, num_heads = 2, 4096, 4096, 32

def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads]