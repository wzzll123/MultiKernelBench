import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, indices, values):
        return x.index_add(dim=0, index=indices, source=values)

def get_inputs():
    x = torch.zeros(1024, 4096)
    indices = torch.randint(0, 1024, (8192,))
    values = torch.rand(8192, 4096)
    return [x, indices, values]

def get_init_inputs():
    return []
