import torch
import torch.nn as nn
import torch_musa
from torch_musa.utils.musa_extension import load_inline

# Define the custom MUSA kernel for layer normalization over the last dimension
layer_norm_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void layer_norm_kernel(const float* x, const float* weight, const float* bias, float* out, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    float mean = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        mean += x_row[i];
    }

    __shared__ float shared[256];
    shared[threadIdx.x] = mean;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    mean = shared[0] / cols;

    float var = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var += diff * diff;
    }
    shared[threadIdx.x] = var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    var = shared[0] / cols;
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_musa(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    auto rows = x.size(0);
    auto cols = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = rows;

    layer_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols, static_cast<float>(eps));

    return out;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_musa(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);"
)

layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    musa_sources=layer_norm_source,
    functions=["layer_norm_musa"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.layer_norm = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm.layer_norm_musa(x, self.ln.weight, self.ln.bias, self.ln.eps)
