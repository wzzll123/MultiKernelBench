import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs the backward pass for embedding with dense gradients.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, indices: torch.Tensor,
                num_weights: int, padding_idx: int = -1, scale_grad_by_freq: bool = False) -> torch.Tensor:
        """
        Computes the gradient for embedding layer with dense backward.

        Args:
            grad_output (torch.Tensor): Gradient of the output.
            indices (torch.Tensor): The indices tensor from forward pass.
            num_weights (int): Number of rows in the embedding weight matrix.
            padding_idx (int, optional): Index of padding token to zero out gradient.
            scale_grad_by_freq (bool, optional): Whether to scale gradients by frequency.

        Returns:
            torch.Tensor: Gradient tensor for embedding weights.
        """
        return torch.ops.aten.embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "24_embedding_dense_backward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        indices_info = inputs[1]
        num_weights_info = inputs[2]
        padding_idx_info = inputs[3]
        scale_grad_by_freq_info = inputs[4]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[grad_output_info["dtype"]]
        
        grad_output = torch.randn(grad_output_info["shape"], dtype=dtype)
        
        index_range = indices_info.get("range", [0, num_weights_info["value"] - 1])
        indices = torch.randint(index_range[0], index_range[1] + 1, tuple(indices_info["shape"]), dtype=torch.int64)
        
        num_weights = num_weights_info["value"]
        padding_idx = padding_idx_info["value"]
        scale_grad_by_freq = scale_grad_by_freq_info["value"]
        
        input_groups.append([grad_output, indices, num_weights, padding_idx, scale_grad_by_freq])
    return input_groups


def get_init_inputs():
    return []
