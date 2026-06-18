import torch
import torch.nn as nn
import torch_musa
from torch_musa.utils.musa_extension import load_inline

# Define the custom MUSA kernel for 1D sum reduction
reduce_sum_source = """
#include <torch/extension.h>
#include <musa_runtime.h>

__global__ void reduce_sum_kernel(const float* x, float* out, int size) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (idx < size) {
        value = x[idx];
    }
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0]);
    }
}

torch::Tensor reduce_sum_musa(torch::Tensor x) {
    auto out = torch::zeros({}, x.options());
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    reduce_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

reduce_sum_cpp_source = (
    "torch::Tensor reduce_sum_musa(torch::Tensor x);"
)

reduce_sum = load_inline(
    name="reduce_sum",
    cpp_sources=reduce_sum_cpp_source,
    musa_sources=reduce_sum_source,
    functions=["reduce_sum_musa"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum.reduce_sum_musa(x)
