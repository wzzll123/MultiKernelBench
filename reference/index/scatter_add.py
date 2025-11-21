import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, idx, updates):
        return x.scatter_add(dim=1, index=idx, src=updates)

def get_inputs():
    x = torch.zeros(32, 4096)
    idx = torch.randint(0, 4096, (32, 1024))
    updates = torch.rand(32, 1024)
    return [x, idx, updates]

def get_init_inputs():
    return []
