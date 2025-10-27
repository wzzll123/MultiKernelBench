import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Cross-modal attention: text queries, visual keys/values.
    """
    def __init__(self, d_model_text=256, d_model_img=512, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model_text, d_model_text)
        self.kv_proj = nn.Linear(d_model_img, d_model_text)
        self.mha = nn.MultiheadAttention(d_model_text, num_heads, batch_first=True)

    def forward(self, text, img):
        q = self.q_proj(text)
        kv = self.kv_proj(img)
        out, _ = self.mha(q, kv, kv)
        return out

batch_size = 4
len_text = 32
len_img = 196

def get_inputs():
    text = torch.randn(batch_size, len_text, 256)
    img = torch.randn(batch_size, len_img, 512)
    return [text, img]

def get_init_inputs():
    return [256, 512, 8]
