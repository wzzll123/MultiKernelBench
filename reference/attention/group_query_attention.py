import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Grouped-Query Attention (GQA)
    -----------------------------
    Like LLaMA-style attention: multiple query heads share a smaller set of key/value heads.
    """
    def __init__(self, d_model=1024, num_heads=16, num_kv_heads=4):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.v_proj = nn.Linear(d_model, d_model * num_kv_heads // num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H, H_kv = self.num_heads, self.num_kv_heads
        head_dim = D // H

        # projections
        Q = self.q_proj(x).view(B, L, H, head_dim)
        K = self.k_proj(x).view(B, L, H_kv, head_dim)
        V = self.v_proj(x).view(B, L, H_kv, head_dim)

        # Expand K/V to match query groups
        if self.group_size > 1:
            K = K.repeat_interleave(self.group_size, dim=2)
            V = V.repeat_interleave(self.group_size, dim=2)

        # attention
        attn = torch.einsum("blhd,bmhd->bh lm", Q, K) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhlm,bmhd->blhd", attn, V).reshape(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------
# Example kernelbench configuration
# ---------------------------------------------------------------------
batch_size = 8
seq_len = 128
d_model = 1024
num_heads = 16
num_kv_heads = 4

def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, num_heads, num_kv_heads]

