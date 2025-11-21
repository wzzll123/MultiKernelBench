import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, scale):
        return x / scale  # broadcast over channels

def get_inputs():
    x = torch.rand(8, 3, 32, 128)
    scale = torch.rand(3, 1, 1)  # broadcast over H and W
    return [x, scale]

def get_init_inputs():
    return []
