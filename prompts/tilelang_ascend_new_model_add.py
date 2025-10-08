import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --- TileLang kernel generator ---
@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M=128, block_N=256, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# --- nn.Module wrapper ---
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape, "Input tensors must have the same shape"
        assert a.dim() == 2, "Expected 2D tensors for this kernel"

        M, N = a.shape
        block_M, block_N = 128, 256
        block_M = min(block_M, M)
        block_N = min(block_N, N)
        func = vec_add(M, N, block_M, block_N)
        return func(a, b)
