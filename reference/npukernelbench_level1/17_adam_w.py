import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs AdamW optimization step using NPU accelerated Adam.
    Pytorch native implemention
    def forward(self, var: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                grad: torch.Tensor, beta1_power: float, beta2_power: float,
                lr: float, beta1: float, beta2: float, epsilon: float,
                use_locking: bool = False, use_nesterov: bool = False):
        orig_dtype = var.dtype

        var_f32 = var.float()
        m_f32 = m.float()
        v_f32 = v.float()
        grad_f32 = grad.float()

        m_new = beta1 * m_f32 + (1 - beta1) * grad_f32
        v_new = beta2 * v_f32 + (1 - beta2) * grad_f32 * grad_f32

        m_hat = m_new / (1 - beta1_power)
        v_hat = v_new / (1 - beta2_power)

        if use_nesterov:
            m_nesterov = beta1 * m_hat + (1 - beta1) * grad_f32 / (1 - beta1_power)
            var_new = var_f32 - lr * m_nesterov / (torch.sqrt(v_hat) + epsilon)
        else:
            var_new = var_f32 - lr * m_hat / (torch.sqrt(v_hat) + epsilon)

        var_out = var_new.to(orig_dtype)
        m_out = m_new.to(m.dtype)
        v_out = v_new.to(v.dtype)

        return var_out, m_out, v_out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                grad: torch.Tensor, beta1_power: float, beta2_power: float,
                lr: float, beta1: float, beta2: float, epsilon: float,
                use_locking: bool = False, use_nesterov: bool = False):
        """
        Applies Adam optimization step on NPU.

        Args:
            var (torch.Tensor): Variable to be updated.
            m (torch.Tensor): First moment estimates.
            v (torch.Tensor): Second moment estimates.
            grad (torch.Tensor): Gradient tensor.
            beta1_power (float): Beta1 power.
            beta2_power (float): Beta2 power.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment.
            beta2 (float): Exponential decay rate for second moment.
            epsilon (float): Small constant for numerical stability.
            use_locking (bool): Whether to use locking.
            use_nesterov (bool): Whether to use Nesterov momentum.

        Returns:
            tuple: Updated (var, m, v).
        """
        import torch_npu
        torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon,
                                  grad, use_locking, use_nesterov, out=(var, m, v))
        return var, m, v


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "17_adam_w.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        
        var_info = inputs[0]
        m_info = inputs[1]
        v_info = inputs[2]
        grad_info = inputs[3]
        beta1_power_info = inputs[4]
        beta2_power_info = inputs[5]
        lr_info = inputs[6]
        beta1_info = inputs[7]
        beta2_info = inputs[8]
        epsilon_info = inputs[9]
        use_locking_info = inputs[10]
        use_nesterov_info = inputs[11]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[var_info["dtype"]]
        
        var = torch.randn(var_info["shape"], dtype=dtype)
        m = torch.randn(m_info["shape"], dtype=dtype)
        v = torch.randn(v_info["shape"], dtype=dtype)
        grad = torch.randn(grad_info["shape"], dtype=dtype)
        
        beta1_power = beta1_power_info["value"]
        beta2_power = beta2_power_info["value"]
        lr = lr_info["value"]
        beta1 = beta1_info["value"]
        beta2 = beta2_info["value"]
        epsilon = epsilon_info["value"]
        use_locking = use_locking_info["value"]
        use_nesterov = use_nesterov_info["value"]
        
        input_groups.append([var, m, v, grad, beta1_power, beta2_power, lr, beta1, beta2, epsilon, use_locking, use_nesterov])
    return input_groups


def get_init_inputs():
    return []
