import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, beta1=0.9, beta2=0.999, lr=1e-3, eps=1e-8, step=1):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.step = step

    def forward(self, param, grad, m, v):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
        m_hat = m / (1 - self.beta1 ** self.step)
        v_hat = v / (1 - self.beta2 ** self.step)
        param = param - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return param

def get_inputs():
    param = torch.rand(512, 2048)
    grad = torch.rand_like(param)
    m = torch.zeros_like(param)
    v = torch.zeros_like(param)
    return [param, grad, m, v]

def get_init_inputs():
    return []
