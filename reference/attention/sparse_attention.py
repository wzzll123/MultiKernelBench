import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Sparse attention: each token attends only to nearby tokens within a fixed window.
    The attention mask is precomputed and registered as a non-trainable buffer.
    """
    def __init__(self, d_model=256, num_heads=4, window_size=32, max_seq_len=4096):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.window_size = window_size
        self.max_seq_len = max_seq_len

        # Precompute sparse mask
        mask = torch.ones(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            start = max(0, i - window_size)
            end = min(max_seq_len, i + window_size)
            mask[i, start:end] = 0
        mask = mask.bool()

        # Register as buffer 
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        attn_mask = self.attn_mask[:seq_len, :seq_len]
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out

batch_size = 16
seq_len = 128
d_model = 512

def get_inputs():
    x = torch.randn(batch_size, seq_len, d_model)
    return [x]

def get_init_inputs():
    return [d_model, 4, 32]
