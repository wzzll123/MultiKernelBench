import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

def get_inputs():
    x = torch.rand(4, 128, 100, 150)
    size = (200, 300)
    return [x, size]

def get_init_inputs():
    return []