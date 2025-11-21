import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def forward(self, x):
        return F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=True)

def get_inputs():
    x = torch.rand(4, 64, 64, 64)
    return [x]


def get_init_inputs():
    return []