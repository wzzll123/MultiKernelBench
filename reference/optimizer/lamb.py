import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, lr=1e-3, eps=1e-6):
        super().__init__()
        self.lr = lr
        self.eps = eps

    def forward(self, param, m, v):
        r = m / (v.sqrt() + self.eps)
        trust_ratio = param.norm(p=2) / (r.norm(p=2) + self.eps)
        param = param - self.lr * trust_ratio * r
        return param

def get_inputs():
    param = torch.rand(512, 4096)
    m = torch.rand_like(param)
    v = torch.abs(torch.rand_like(param)) + 1e-3
    return [param, m, v]

def get_init_inputs():
    return []
