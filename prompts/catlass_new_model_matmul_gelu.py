project_json_src='''
[
    {
        "op": "MatmulGeluCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "a",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float16"
                ]
            },
            {
                "name": "b",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float16"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "c",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float16"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, K);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulGeluCustom, MatmulGeluCustomTilingData)
}
"""

host_operator_src="""
#include "matmul_gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MatmulGeluCustomTilingData tiling;
  const gert::StorageShape* a_shape = context->GetInputShape(0);
  const gert::StorageShape* b_shape = context->GetInputShape(1);

  uint32_t M = a_shape->GetStorageShape().GetDim(0);
  uint32_t K = a_shape->GetStorageShape().GetDim(1);
  uint32_t N = b_shape->GetStorageShape().GetDim(1);

  auto aic_core_num = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
  context->SetBlockDim(aic_core_num);

  tiling.set_M(M);
  tiling.set_N(N);
  tiling.set_K(K);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();
  size_t required_workspace_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  size_t* current_workspace = context->GetWorkspaceSizes(1);
  current_workspace[0] = required_workspace_bytes + sys_workspace_size;

  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* a_shape = context->GetInputShape(0);
    const gert::Shape* b_shape = context->GetInputShape(1);
    gert::Shape* c_shape = context->GetOutputShape(0);
    if (a_shape->GetDimNum() >= 2 && b_shape->GetDimNum() >= 2) {
        c_shape->SetDimNum(2);
        c_shape->SetDim(0, a_shape->GetDim(0));
        c_shape->SetDim(1, b_shape->GetDim(1));
    } else {
        *c_shape = *a_shape;
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulGeluCustom : public OpDef {
public:
    explicit MatmulGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulGeluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

using namespace Catlass;

using ArchTag = Arch::AtlasA2;
using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;
using LayoutD = layout::RowMajor;
using ElementA = half;
using ElementB = half;
using ElementC = float;
using ElementD = half;

using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1Shape = GemmShape<128, 256, 256>;
using L0Shape = GemmShape<128, 256, 64>;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1Shape, L0Shape, AType, BType, CType>;

constexpr uint32_t matmulReady = 0;

class KernelCube {
public:
    __aicore__ inline KernelCube() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, uint32_t M, uint32_t N, uint32_t K) {
        this->M = M;
        this->N = N;
        this->K = K;

        constexpr uint32_t tileM = (uint32_t)L1Shape::M;
        constexpr uint32_t tileN = (uint32_t)L1Shape::N;
        uint32_t tileCountM = (M + tileM - 1) / tileM;
        tileCountN = (N + tileN - 1) / tileN;
        totalTiles = tileCountM * tileCountN;

        gmA.SetGlobalBuffer((__gm__ ElementA *)a);
        gmB.SetGlobalBuffer((__gm__ ElementB *)b);
        gmC.SetGlobalBuffer((__gm__ ElementC *)workspace);
    }

    __aicore__ inline void Process() {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        LayoutA layoutA(M, K);
        LayoutB layoutB(K, N);
        LayoutC layoutC(M, N);
        constexpr uint32_t tileM = (uint32_t)L1Shape::M;
        constexpr uint32_t tileN = (uint32_t)L1Shape::N;

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < totalTiles; loopIdx += AscendC::GetBlockNum()) {
            uint32_t blockM = loopIdx / tileCountN;
            uint32_t blockN = loopIdx % tileCountN;
            uint32_t offsetM = blockM * tileM;
            uint32_t offsetN = blockN * tileN;
            uint32_t actualM = ((M - offsetM) > tileM) ? tileM : (M - offsetM);
            uint32_t actualN = ((N - offsetN) > tileN) ? tileN : (N - offsetN);
            GemmCoord actualBlockShape{actualM, actualN, K};

            MatrixCoord offsetA{offsetM, 0};
            MatrixCoord offsetB{0, offsetN};
            MatrixCoord offsetC{offsetM, offsetN};

            blockMmad(gmA[layoutA.GetOffset(offsetA)], layoutA,
                      gmB[layoutB.GetOffset(offsetB)], layoutB,
                      gmC[layoutC.GetOffset(offsetC)], layoutC,
                      actualBlockShape);

            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(matmulReady);
        }
    }

private:
    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementB> gmB;
    AscendC::GlobalTensor<ElementC> gmC;

    uint32_t M, N, K, totalTiles, tileCountN;
};

class KernelVector {
public:
    __aicore__ inline KernelVector() {}

    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K) {
        this->M = M;
        this->N = N;

        constexpr uint32_t tileM = (uint32_t)L1Shape::M;
        constexpr uint32_t tileN = (uint32_t)L1Shape::N;
        uint32_t tileCountM = (M + tileM - 1) / tileM;
        tileCountN = (N + tileN - 1) / tileN;
        totalTiles = tileCountM * tileCountN;

        aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        subcoreIndex = AscendC::GetSubBlockIdx();

        gmC.SetGlobalBuffer((__gm__ ElementC *)workspace);
        gmD.SetGlobalBuffer((__gm__ ElementD *)c);

        constexpr uint32_t VEC_NUM = 2;
        computeLength = (L1Shape::M / VEC_NUM) * L1Shape::N;

        pipe.InitBuffer(inQueueC, 1, computeLength * sizeof(ElementC));
        pipe.InitBuffer(outQueueD, 1, computeLength * sizeof(ElementD));
        pipe.InitBuffer(tmpBuf, computeLength * sizeof(ElementC));
    }

    __aicore__ inline void Process() {
        constexpr uint32_t tileM = (uint32_t)L1Shape::M;
        constexpr uint32_t tileN = (uint32_t)L1Shape::N;
        for (uint32_t loopIdx = aicoreIndex; loopIdx < totalTiles; loopIdx += AscendC::GetBlockNum()) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(matmulReady);

            uint32_t blockM = loopIdx / tileCountN;
            uint32_t blockN = loopIdx % tileCountN;
            uint32_t offsetM = blockM * tileM;
            uint32_t offsetN = blockN * tileN;
            uint32_t actualM = ((M - offsetM) > tileM) ? tileM : (M - offsetM);
            uint32_t actualN = ((N - offsetN) > tileN) ? tileN : (N - offsetN);

            uint32_t subM = (actualM + AscendC::GetSubBlockNum() - 1) / AscendC::GetSubBlockNum();
            uint32_t subOffsetM = subcoreIndex * subM;
            currActualSubM = (subOffsetM + subM <= actualM) ? subM : (actualM > subOffsetM ? actualM - subOffsetM : 0);

            if (currActualSubM > 0) {
                currGmRow = offsetM + subOffsetM;
                currGmCol = offsetN;
                currActualN = actualN;

                CopyIn();
                Compute();
                CopyOut();
            }
        }
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<ElementC> ubC = inQueueC.AllocTensor<ElementC>();
        auto actualSubblockShape = MakeCoord(currActualSubM, currActualN);
        auto ubTileStride = MakeCoord((int64_t)L1Shape::N, 1L);
        layout::RowMajor layoutComputeUb(actualSubblockShape, ubTileStride);
        LayoutC layoutC(M, N);

        copyGmToUbC(ubC, gmC[layoutC.GetOffset(MakeCoord(currGmRow, currGmCol))],
                    layoutComputeUb, layoutC.GetTileLayout(actualSubblockShape));
        inQueueC.EnQue(ubC);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<ElementC> ubC = inQueueC.DeQue<ElementC>();
        AscendC::LocalTensor<ElementD> ubD = outQueueD.AllocTensor<ElementD>();
        AscendC::LocalTensor<ElementC> ubCompute = tmpBuf.Get<ElementC>();

        const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
        const float TANH_APPROX_FACTOR = 1.0f / 0.044715f;

        AscendC::Mul(ubCompute, ubC, ubC, computeLength);
        AscendC::Mul(ubCompute, ubCompute, ubC, computeLength);
        AscendC::Axpy(ubCompute, ubC, TANH_APPROX_FACTOR, computeLength);
        AscendC::Muls(ubCompute, ubCompute, NEG_SQRT_EIGHT_OVER_PI, computeLength);
        AscendC::Exp(ubCompute, ubCompute, computeLength);
        AscendC::Adds(ubCompute, ubCompute, 1.0f, computeLength);
        AscendC::Div(ubCompute, ubC, ubCompute, computeLength);
        AscendC::Cast(ubD, ubCompute, AscendC::RoundMode::CAST_NONE, computeLength);

        outQueueD.EnQue(ubD);
        inQueueC.FreeTensor(ubC);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<ElementD> ubD = outQueueD.DeQue<ElementD>();
        auto actualSubblockShape = MakeCoord(currActualSubM, currActualN);
        auto ubTileStride = MakeCoord((int64_t)L1Shape::N, 1L);
        layout::RowMajor layoutComputeUb(actualSubblockShape, ubTileStride);
        LayoutD layoutD(M, N);

        copyUbToGmD(gmD[layoutD.GetOffset(MakeCoord(currGmRow, currGmCol))], ubD,
                    layoutD.GetTileLayout(actualSubblockShape), layoutComputeUb);
        outQueueD.FreeTensor(ubD);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueC;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueD;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementD> gmD;

    using CopyGmToUbC = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<ElementC, LayoutC>>;
    using CopyUbToGmD = Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementD, LayoutD>>;
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;

    uint32_t M, N, totalTiles, tileCountN;
    uint32_t aicoreIndex, subcoreIndex, computeLength;
    uint32_t currGmRow, currGmCol, currActualSubM, currActualN;
};

extern "C" __global__ __aicore__
void matmul_gelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);

    if ASCEND_IS_AIC {
        KernelCube opCube;
        opCube.Init(a, b, workspace, t.M, t.N, t.K);
        opCube.Process();
    } else if ASCEND_IS_AIV {
        KernelVector opVector;
        opVector.Init(workspace, c, t.M, t.N, t.K);
        opVector.Process();
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor matmul_gelu_impl_npu(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul_gelu_custom only supports 2D tensors.");
    TORCH_CHECK(a.size(1) == b.size(0), "matmul_gelu_custom: inner dimensions must match.");
    at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());
    EXEC_NPU_CMD(aclnnMatmulGeluCustom, a, b, out);
    return out;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_gelu_custom", &matmul_gelu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_gelu_custom", &matmul_gelu_impl_npu, "matmul and gelu");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return custom_ops_lib.matmul_gelu_custom(a, b)
'''
