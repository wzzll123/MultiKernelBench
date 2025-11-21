import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def forward(self, param, grad, v):
        v = self.alpha * v + (1 - self.alpha) * grad.pow(2)
        param = param - self.lr * grad / (v.sqrt() + self.eps)
        return param

def get_inputs():
    param = torch.rand(2048, 2048)
    grad = torch.rand_like(param)
    v = torch.zeros_like(param)
    return [param, grad, v]

def get_init_inputs():
    return []
