project_json_src='''
[
    {
        "op": "AddCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
"""

host_operator_src="""
#include <algorithm>
#include <cstdint>
#include "add_custom_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t MIN_ELEMS_PER_CORE = 1024;
const uint32_t PIPELINE_DEPTH = 2;
const uint32_t BUFFER_NUM = 3;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    AddCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint32_t blockDim = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    if (blockDim == 0) {
        blockDim = 1;
    }
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    uint32_t tileSize = static_cast<uint32_t>(ubSize / PIPELINE_DEPTH / BUFFER_NUM);
    if (tileSize == 0) {
        tileSize = 1024;
    }

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_tileSize(tileSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AddCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t PIPELINE_DEPTH = 2;
 
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t blockLength, uint32_t tileSize)
    {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileSize = tileSize;
        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        this->currentBlockLength = this->totalLength - blockOffset;
        if (this->currentBlockLength > this->blockLength) {
            this->currentBlockLength = this->blockLength;
        }
        this->elementNumPerTile = this->tileSize / sizeof(DTYPE_X);
        if (this->elementNumPerTile == 0) {
            this->elementNumPerTile = 1;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockOffset, this->currentBlockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + blockOffset, this->currentBlockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + blockOffset, this->currentBlockLength);
        pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, this->tileSize);
        pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, this->tileSize);
        pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, this->tileSize);
    }
    __aicore__ inline void Process()
    {
        uint32_t tileNum = this->currentBlockLength / this->elementNumPerTile;
        uint32_t tailTileElementNum = this->currentBlockLength - tileNum * this->elementNumPerTile;
        for (uint32_t i = 0; i < tileNum; i++) {
            ProcessTile(i * this->elementNumPerTile, this->elementNumPerTile);
        }
        if (tailTileElementNum > 0) {
            ProcessTile(tileNum * this->elementNumPerTile, tailTileElementNum);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t elementNum)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementNum * sizeof(DTYPE_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DTYPE_X> xPadParams{false, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DTYPE_Y> yPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, xPadParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, yPadParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        xLocal = inQueueX.DeQue<DTYPE_X>();
        yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, elementNum);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t currentBlockLength;
    uint32_t tileSize;
    uint32_t elementNumPerTile;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_custom", &add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &add_custom_impl_npu, "x + y");
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
        return custom_ops_lib.add_custom(a, b)
'''
