import torch
import torch.nn as nn
import torch_musa
from torch_musa.utils.musa_extension import load_inline

# Define the custom MUSA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = x[idx];
        out[idx] = value > 0.0f ? value : value * negative_slope;
    }
}

torch::Tensor leaky_relu_musa(torch::Tensor x, double negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), static_cast<float>(negative_slope), size);

    return out;
}
"""

leaky_relu_cpp_source = (
    "torch::Tensor leaky_relu_musa(torch::Tensor x, double negative_slope);"
)

leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_cpp_source,
    musa_sources=leaky_relu_source,
    functions=["leaky_relu_musa"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu.leaky_relu_musa(x, self.negative_slope)
