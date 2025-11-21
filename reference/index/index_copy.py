import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, indices, src):
        return x.index_copy(0, indices, src)

def get_inputs():
    x = torch.zeros(8192, 1024)
    indices = torch.randint(0, 8192, (2048,))
    src = torch.rand(2048, 1024)
    return [x, indices, src]

def get_init_inputs():
    return []
