import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module):
    """
    Simulates autoregressive inference attention with KV caching.

    Q: (batch_size, q_len, d_model)
    K/V: (batch_size, kv_len, d_model)
    Output: (batch_size, q_len, d_model)
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K_cache, V_cache):
        output = F.scaled_dot_product_attention(Q, K_cache, V_cache)
        return output
    
batch_size = 1
q_len = 1
kv_len = 2048
d_model = 4096

def get_inputs():
    Q = torch.randn(batch_size, q_len, d_model)
    K = torch.randn(batch_size, kv_len, d_model)
    V = torch.randn(batch_size, kv_len, d_model)
    return [Q, K, V]

def get_init_inputs():
    return []