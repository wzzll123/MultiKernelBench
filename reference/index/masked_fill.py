import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, mask):
        return x.masked_fill(mask, float('-inf'))

def get_inputs():
    x = torch.rand(64, 512, 512)
    mask = torch.randint(0, 2, (64, 512, 512), dtype=torch.bool)
    return [x, mask]

def get_init_inputs():
    return []
