import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, idx, updates):
        return x.scatter(dim=1, index=idx, src=updates)

def get_inputs():
    x = torch.zeros(64, 8192)
    idx = torch.randint(0, 8192, (64, 4096))
    updates = torch.rand_like(idx, dtype=torch.float)
    return [x, idx, updates]

def get_init_inputs():
    return []
