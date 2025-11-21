import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a * b  # broadcast multiply

def get_inputs():
    a = torch.rand(4, 1, 2048)
    b = torch.rand(1, 4, 2048)
    return [a, b]

def get_init_inputs():
    return []
