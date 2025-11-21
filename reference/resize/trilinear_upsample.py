import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

def get_inputs():
    x = torch.rand(2, 16, 32, 32, 32)
    return [x]

def get_init_inputs():
    return []