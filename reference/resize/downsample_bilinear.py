import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, size=(60, 80), mode='bilinear', align_corners=False)

def get_inputs():
    x = torch.rand(4, 128, 240, 320)
    return [x]

def get_init_inputs():
    return []