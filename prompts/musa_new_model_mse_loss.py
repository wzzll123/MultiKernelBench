import torch
import torch.nn as nn
import torch_musa
from torch_musa.utils.musa_extension import load_inline

# Define the custom MUSA kernel for squared difference
squared_diff_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void squared_diff_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        out[idx] = diff * diff;
    }
}

torch::Tensor squared_diff_musa(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    squared_diff_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

squared_diff_cpp_source = (
    "torch::Tensor squared_diff_musa(torch::Tensor predictions, torch::Tensor targets);"
)

squared_diff = load_inline(
    name="squared_diff",
    cpp_sources=squared_diff_cpp_source,
    musa_sources=squared_diff_source,
    functions=["squared_diff_musa"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.squared_diff = squared_diff

    def forward(self, predictions, targets):
        return torch.mean(self.squared_diff.squared_diff_musa(predictions, targets))
