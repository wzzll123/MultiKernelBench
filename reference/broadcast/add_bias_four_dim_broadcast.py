import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bias):
        return x + bias.view(1, -1, 1, 1)

def get_inputs():
    x = torch.rand(8, 512, 32, 32)
    bias = torch.rand(512)
    return [x, bias]

def get_init_inputs():
    return []
