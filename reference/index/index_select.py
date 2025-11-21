import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, indices):
        return torch.index_select(x, dim=1, index=indices)

def get_inputs():
    x = torch.rand(256, 8192)
    indices = torch.randint(0, 8192, (2048,))
    return [x, indices]

def get_init_inputs():
    return []
