import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Causal (decoder) self-attention with an upper-triangular mask.
    Mask is precomputed and registered as a non-trainable buffer.
    """
    def __init__(self, d_model=256, num_heads=4, max_seq_len=4096):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.max_seq_len = max_seq_len

        # Precompute causal mask (1 for disallowed positions)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        attn_mask = self.causal_mask[:seq_len, :seq_len]
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out

batch_size = 32
seq_len = 64
d_model = 512

def get_inputs():
    x = torch.rand(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, 4]
