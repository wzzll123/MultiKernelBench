import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, momentum=0.9, lr=1e-2):
        super().__init__()
        self.momentum = momentum
        self.lr = lr

    def forward(self, param, grad, velocity):
        velocity = self.momentum * velocity + grad
        param = param - self.lr * velocity
        return param

def get_inputs():
    param = torch.rand(1024, 4096)
    grad = torch.rand(1024, 4096)
    velocity = torch.zeros_like(param)
    return [param, grad, velocity]

def get_init_inputs():
    return []
