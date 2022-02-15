// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QLinearLongformerAttention final : public CudaKernel, public LongformerAttentionBase {
 public:
  QLinearLongformerAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_compact_memory_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
