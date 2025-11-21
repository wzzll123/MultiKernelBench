import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)

def get_inputs():
    x = torch.rand(8, 256, 64, 64)
    return [x]

def get_init_inputs():
    return []
