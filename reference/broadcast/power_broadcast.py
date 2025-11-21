import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, exponent):
        return x ** exponent

def get_inputs():
    x = torch.abs(torch.rand(4, 2048)) + 1e-2
    exponent = torch.full((2048,), 0.5,)
    return [x, exponent]

def get_init_inputs():
    return []
