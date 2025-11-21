import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, idx):
        return torch.gather(x, dim=1, index=idx)

def get_inputs():
    x = torch.rand(128, 8192)
    idx = torch.randint(0, 8192, (128, 4096))
    return [x, idx]

def get_init_inputs():
    return []
