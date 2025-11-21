import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=4.0, mode='nearest')

def get_inputs():
    x = torch.rand(16, 128, 32, 32)
    return [x]

def get_init_inputs():
    return []