import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, lr=1e-2, eps=1e-10):
        super().__init__()
        self.lr = lr
        self.eps = eps

    def forward(self, param, grad, accum):
        accum = accum + grad.pow(2)
        param = param - self.lr * grad / (accum.sqrt() + self.eps)
        return param

def get_inputs():
    param = torch.rand(1024, 4096)
    grad = torch.rand_like(param)
    accum = torch.zeros_like(param)
    return [param, grad, accum]

def get_init_inputs():
    return []
