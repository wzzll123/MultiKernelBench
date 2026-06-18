import torch
import torch.nn as nn
import torch_musa
from torch_musa.utils.musa_extension import load_inline

# Define the custom MUSA kernel for adding bias after matmul
add_bias_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void add_bias_kernel(const float* x, const float* bias, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        int col = idx % cols;
        out[idx] = x[idx] + bias[col];
    }
}

torch::Tensor add_bias_musa(torch::Tensor x, torch::Tensor bias) {
    auto rows = x.size(0);
    auto cols = x.size(1);
    auto out = torch::zeros_like(x);
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    add_bias_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

    return out;
}
"""

add_bias_cpp_source = (
    "torch::Tensor add_bias_musa(torch::Tensor x, torch::Tensor bias);"
)

add_bias = load_inline(
    name="add_bias",
    cpp_sources=add_bias_cpp_source,
    musa_sources=add_bias_source,
    functions=["add_bias_musa"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.add_bias = add_bias

    def forward(self, a, b, bias):
        x = torch.matmul(a, b)
        return self.add_bias.add_bias_musa(x.float(), bias.float())
