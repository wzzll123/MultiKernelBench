import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bias):
        return x - bias

def get_inputs():
    x = torch.rand(8, 4096)
    bias = torch.rand(4096)
    return [x, bias]

def get_init_inputs():
    return []