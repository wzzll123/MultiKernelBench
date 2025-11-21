import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cond, x, y):
        return torch.where(cond, x, y)

def get_inputs():
    cond = torch.randint(0, 2, (1, 512, 1), dtype=torch.bool)
    x = torch.rand(64, 512, 64)
    y = torch.rand(64, 512, 64)
    return [cond, x, y]

def get_init_inputs():
    return []
