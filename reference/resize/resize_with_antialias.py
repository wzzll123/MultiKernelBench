import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False, antialias=True)

def get_inputs():
    x = torch.rand(8, 3, 256, 256)
    return [x]

def get_init_inputs():
    return []