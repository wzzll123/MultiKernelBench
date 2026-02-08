import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b):
        return F.gelu(torch.matmul(a, b))


def get_inputs():
    a = torch.randn(1024, 256, dtype=torch.float16)
    b = torch.randn(256, 640, dtype=torch.float16)
    return [a, b]


def get_init_inputs():
    return []
