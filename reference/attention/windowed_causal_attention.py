import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Sliding-window causal attention (local attention).
    Each token attends only to a fixed window of previous tokens.
    Used in Longformer, MPT, and streaming LLMs.
    """
    def __init__(self, d_model=1024, num_heads=16, window_size=256):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.window_size = window_size

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d_h = D // H

        Q = self.q_proj(x).view(B, N, H, d_h)
        K = self.k_proj(x).view(B, N, H, d_h)
        V = self.v_proj(x).view(B, N, H, d_h)

        attn_out = torch.zeros_like(Q)
        for i in range(0, N):
            start = max(0, i - self.window_size)
            Qi = Q[:, i:i+1]
            Ki = K[:, start:i+1]
            Vi = V[:, start:i+1]
            scores = torch.einsum("bqhd,bkhd->bhqk", Qi, Ki) / math.sqrt(d_h)
            weights = F.softmax(scores, dim=-1)
            attn_out[:, i:i+1] = torch.einsum("bhqk,bkhd->bqhd", weights, Vi)
        return self.out_proj(attn_out.reshape(B, N, D))

batch_size, seq_len, d_model, num_heads, window_size = 4, 2048, 1024, 16, 256

def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, window_size]
